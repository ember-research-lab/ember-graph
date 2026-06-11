//! spectral — normalized-Laplacian eigendecomposition for the knowledge graph (v0.2 item 5).
//!
//! Computes spectral properties of the **symmetric normalized Laplacian**
//! `L = I − D^(−1/2) A D^(−1/2)` of the graph's undirected symmetrization.
//!
//! **Graph model used here:**
//! - Directed edges are symmetrized: if an edge `u→v` with score `w₁` and `v→u` with score `w₂`
//!   both exist, the undirected weight is `w₁ + w₂`. If only one direction exists, the weight is
//!   that direction's score.
//! - Edge weight = `edge.confidence.score()` cast to f64.
//! - Self-loops are ignored.
//! - Nodes are operated on in **sorted [`NodeId`] order** (deterministic). A GNN consumer can
//!   re-align tensor rows via [`NodeIndex::u64_of`](crate::NodeIndex::u64_of).
//!
//! **Isolated nodes** (degree 0) have an undefined `D^(−1/2)`, but the symmetric normalized Laplacian
//! resolves this naturally: an isolated node's row/column is zero off-diagonal with diagonal 1, so it
//! is an eigenvector with **eigenvalue 1** (no NaN). This is a known quirk of the *normalized* (vs
//! unnormalized) Laplacian — an isolated node does **not** add a zero eigenvalue, so it does **not**
//! by itself drive algebraic connectivity to 0. Only two or more mutually-unreachable components that
//! each contain an edge produce extra zero eigenvalues. A caller that wants isolated nodes treated as
//! "disconnected" should detect them directly (degree 0), not infer it from the Fiedler value.
//!
//! **Dense-only limitation:** The full n×n Laplacian is materialized and decomposed via `faer`'s
//! dense symmetric eigendecomposition. This is fine for modest graphs (up to a few thousand
//! nodes), but will be slow and memory-hungry for very large ones. A future version may use a
//! sparse Lanczos solver.
//!
//! **Determinism:** Eigenvectors are sign-ambiguous by convention. We canonicalize each vector by
//! forcing the entry of largest absolute value to be positive. Ties are broken by the sorted node
//! order (the first maximum-magnitude entry wins). This makes the outputs stable across
//! runs/platforms for the same graph topology.

use std::collections::HashMap;

use faer::{linalg::solvers::SelfAdjointEigen, Mat, Side};

use crate::{Graph, NodeId};

// ── Public error type ────────────────────────────────────────────────────────

/// Errors returned by spectral computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpectralError {
    /// The graph has no live nodes — the Laplacian is 0×0.
    EmptyGraph,
    /// Not enough nodes for the requested computation.
    TooFewNodes {
        /// How many nodes the graph actually has.
        have: usize,
        /// How many are required.
        need: usize,
    },
    /// The eigendecomposition did not converge. (Rare for symmetric matrices.)
    NoConvergence,
}

impl std::fmt::Display for SpectralError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectralError::EmptyGraph => write!(f, "spectral: graph has no nodes"),
            SpectralError::TooFewNodes { have, need } => write!(
                f,
                "spectral: need at least {need} node(s), graph has {have}"
            ),
            SpectralError::NoConvergence => {
                write!(f, "spectral: eigendecomposition did not converge")
            }
        }
    }
}

impl std::error::Error for SpectralError {}

// ── Public result type ───────────────────────────────────────────────────────

/// The output of [`spectral_embedding`]: per-node coordinates in the `k`-dimensional spectral
/// space defined by the `k` smallest non-trivial eigenvectors of the normalized Laplacian.
///
/// `nodes[i]` is the node id; `coords[i]` is its `k`-dimensional coordinate vector;
/// `eigenvalues[j]` is the j-th (non-trivial, non-decreasing) eigenvalue.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralEmbedding {
    /// Nodes in sorted [`NodeId`] order (the same order as the Laplacian rows).
    pub nodes: Vec<NodeId>,
    /// The `k` non-trivial eigenvalues, non-decreasing.
    pub eigenvalues: Vec<f64>,
    /// `coords[i]` is node `i`'s `k`-dimensional coordinate vector.
    pub coords: Vec<Vec<f64>>,
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Build the sorted node list and the symmetrized adjacency map.
///
/// Returns `(sorted_node_ids, undirected_weights)` where `undirected_weights[(i, j)]` is the
/// combined edge weight between node index `i` and `j` (i ≠ j, i < j for the stored key, but
/// we look up both directions when building the Laplacian).
fn build_adjacency(graph: &Graph) -> (Vec<NodeId>, HashMap<(usize, usize), f64>) {
    let mut nodes: Vec<NodeId> = graph.nodes.keys().cloned().collect();
    nodes.sort();

    let pos: HashMap<&NodeId, usize> = nodes.iter().enumerate().map(|(i, id)| (id, i)).collect();

    let mut weights: HashMap<(usize, usize), f64> = HashMap::new();

    for edges in graph.out.values() {
        for e in edges {
            if e.source == e.target {
                continue; // skip self-loops
            }
            let Some(&i) = pos.get(&e.source) else {
                continue;
            };
            let Some(&j) = pos.get(&e.target) else {
                continue;
            };
            // Symmetrize: accumulate in canonical (min, max) key.
            let key = (i.min(j), i.max(j));
            *weights.entry(key).or_insert(0.0) += e.confidence.score() as f64;
        }
    }

    (nodes, weights)
}

/// Build the dense n×n symmetric normalized Laplacian matrix.
///
/// `L[i][j] = -w_{ij} / sqrt(d_i * d_j)` for i ≠ j with an edge,
/// `L[i][i] = 1` for nodes with positive degree, and
/// `L[i][i] = 1` for isolated nodes (degree 0 — see module doc).
fn build_laplacian(n: usize, weights: &HashMap<(usize, usize), f64>) -> Mat<f64> {
    // Compute degree for each node.
    let mut degree = vec![0.0f64; n];
    for (&(i, j), &w) in weights {
        degree[i] += w;
        degree[j] += w;
    }

    let mut lap = Mat::<f64>::zeros(n, n);

    // Diagonal: 1 for all nodes (both isolated and connected).
    for i in 0..n {
        lap[(i, i)] = 1.0;
    }

    // Off-diagonal: -w_{ij} / sqrt(d_i * d_j) for each edge.
    for (&(i, j), &w) in weights {
        let di = degree[i];
        let dj = degree[j];
        if di > 0.0 && dj > 0.0 {
            let val = -w / (di * dj).sqrt();
            lap[(i, j)] = val;
            lap[(j, i)] = val;
        }
    }

    lap
}

/// Run the dense symmetric eigendecomposition and return `(eigenvalues, U_matrix)`.
///
/// Eigenvalues are returned in ascending order (faer's self-adjoint EVD guarantees this).
fn decompose(lap: Mat<f64>) -> Result<(Vec<f64>, Mat<f64>), SpectralError> {
    let evd = SelfAdjointEigen::<f64>::new(lap.as_ref(), Side::Lower)
        .map_err(|_| SpectralError::NoConvergence)?;

    let s = evd.S();
    let eigenvalues: Vec<f64> = s.column_vector().iter().copied().collect();

    // `evd.U()` is n×n; column j is the eigenvector for eigenvalue j.
    // We need to own the data: copy it into a plain Mat<f64>.
    let u_ref = evd.U();
    let n = u_ref.nrows();
    let mut u = Mat::<f64>::zeros(n, n);
    for row in 0..n {
        for col in 0..n {
            u[(row, col)] = u_ref[(row, col)];
        }
    }

    Ok((eigenvalues, u))
}

/// Canonicalize the sign of an eigenvector (in place, as a `Vec<f64>`) by forcing the entry of
/// largest absolute value to be positive. This makes sign-ambiguous eigenvectors deterministic
/// across runs.
fn canonicalize_sign(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    // Find the entry with the largest absolute value; if it is negative, flip the whole vector.
    let max_abs =
        v.iter()
            .copied()
            .map(f64::abs)
            .enumerate()
            .fold(
                (0usize, 0.0f64),
                |(bi, bv), (i, av)| {
                    if av > bv {
                        (i, av)
                    } else {
                        (bi, bv)
                    }
                },
            );
    if v[max_abs.0] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// Internal return type for [`spectral_core`]: sorted node list, eigenvalues, and eigenvector
/// matrix U (columns are eigenvectors, ascending eigenvalue order).
type SpectralCore = (Vec<NodeId>, Vec<f64>, Mat<f64>);

/// Common setup: build adjacency + Laplacian + decompose, returning
/// `(nodes, eigenvalues, U_matrix_columns_are_eigenvectors)`.
fn spectral_core(graph: &Graph) -> Result<SpectralCore, SpectralError> {
    let n = graph.node_count();
    if n == 0 {
        return Err(SpectralError::EmptyGraph);
    }
    let (nodes, weights) = build_adjacency(graph);
    let lap = build_laplacian(n, &weights);
    let (eigenvalues, u) = decompose(lap)?;
    Ok((nodes, eigenvalues, u))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// The **algebraic connectivity** (Fiedler value) — the 2nd-smallest eigenvalue of the
/// symmetric normalized Laplacian.
///
/// - ≈ 0 iff the graph is disconnected (or has isolated nodes).
/// - Larger values mean better connectivity.
///
/// Requires at least 2 nodes.
pub fn algebraic_connectivity(graph: &Graph) -> Result<f64, SpectralError> {
    let n = graph.node_count();
    if n == 0 {
        return Err(SpectralError::EmptyGraph);
    }
    if n < 2 {
        return Err(SpectralError::TooFewNodes { have: n, need: 2 });
    }
    let (_, eigenvalues, _) = spectral_core(graph)?;
    // eigenvalues are sorted ascending by faer; index 1 is the 2nd-smallest.
    Ok(eigenvalues[1])
}

/// The **Fiedler vector** — the eigenvector corresponding to the 2nd-smallest eigenvalue.
///
/// Its sign gives a natural bipartition: nodes with positive coordinates are in one cluster,
/// negative in the other. Sign is canonicalized (largest-magnitude entry is positive).
///
/// Returns a `Vec<(NodeId, f64)>` in sorted [`NodeId`] order.
///
/// Requires at least 2 nodes.
pub fn fiedler_vector(graph: &Graph) -> Result<Vec<(NodeId, f64)>, SpectralError> {
    let n = graph.node_count();
    if n == 0 {
        return Err(SpectralError::EmptyGraph);
    }
    if n < 2 {
        return Err(SpectralError::TooFewNodes { have: n, need: 2 });
    }
    let (nodes, _, u) = spectral_core(graph)?;
    // Column 1 of U is the Fiedler vector (2nd-smallest eigenvalue).
    let mut vec: Vec<f64> = (0..n).map(|i| u[(i, 1)]).collect();
    canonicalize_sign(&mut vec);
    Ok(nodes.into_iter().zip(vec).collect())
}

/// The **spectral embedding** — the `k` smallest non-trivial eigenvectors as per-node coordinates.
///
/// "Non-trivial" skips the 1st eigenvector (the near-constant one with eigenvalue ≈ 0).
/// So the `k` vectors correspond to eigenvalue indices 1..=k (0-indexed).
///
/// Requires at least `k + 1` nodes (to have `k` non-trivial eigenvectors).
///
/// # Returns
///
/// A [`SpectralEmbedding`] where:
/// - `nodes` is in sorted [`NodeId`] order.
/// - `eigenvalues[j]` is the j-th non-trivial eigenvalue (non-decreasing).
/// - `coords[i][j]` is node `i`'s coordinate on axis `j`.
pub fn spectral_embedding(graph: &Graph, k: usize) -> Result<SpectralEmbedding, SpectralError> {
    let n = graph.node_count();
    if n == 0 {
        return Err(SpectralError::EmptyGraph);
    }
    // We need at least k+1 nodes: 1 trivial + k non-trivial eigenvectors.
    if n < k + 1 {
        return Err(SpectralError::TooFewNodes {
            have: n,
            need: k + 1,
        });
    }
    let (nodes, eigenvalues, u) = spectral_core(graph)?;

    // Non-trivial eigenvectors: columns 1..=k of U.
    let selected_eigenvalues: Vec<f64> = eigenvalues[1..=k].to_vec();

    // Build sign-canonicalized eigenvectors.
    let mut vecs: Vec<Vec<f64>> = (1..=k)
        .map(|col| {
            let mut v: Vec<f64> = (0..n).map(|row| u[(row, col)]).collect();
            canonicalize_sign(&mut v);
            v
        })
        .collect();

    // coords[i] = node i's k coordinates (one per eigenvector).
    let coords: Vec<Vec<f64>> = (0..n)
        .map(|i| vecs.iter_mut().map(|v| v[i]).collect())
        .collect();

    Ok(SpectralEmbedding {
        nodes,
        eigenvalues: selected_eigenvalues,
        coords,
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Confidence, Edge, Graph, Node, NodeId};

    fn node(id: &str) -> Node {
        Node {
            id: NodeId(id.into()),
            label: id.into(),
            kind: "test".into(),
            source: None,
        }
    }

    fn edge(s: &str, t: &str) -> Edge {
        Edge {
            source: NodeId(s.into()),
            target: NodeId(t.into()),
            relation: "r".into(),
            confidence: Confidence::Extracted, // score = 1.0
        }
    }

    /// Build K₂ (two nodes, one undirected edge weight 1).
    /// Normalized Laplacian eigenvalues are exactly {0, 2}.
    fn k2() -> Graph {
        let mut g = Graph::new();
        g.add_node(node("a"));
        g.add_node(node("b"));
        g.add_edge(edge("a", "b"));
        g.add_edge(edge("b", "a")); // symmetrize explicitly
        g
    }

    /// Two disconnected triangles: {a,b,c} and {d,e,f}, each fully connected.
    fn two_triangles() -> Graph {
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d", "e", "f"] {
            g.add_node(node(id));
        }
        // Triangle 1
        for (s, t) in [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("a", "c"),
            ("c", "a"),
        ] {
            g.add_edge(edge(s, t));
        }
        // Triangle 2
        for (s, t) in [
            ("d", "e"),
            ("e", "d"),
            ("e", "f"),
            ("f", "e"),
            ("d", "f"),
            ("f", "d"),
        ] {
            g.add_edge(edge(s, t));
        }
        g
    }

    /// Barbell: two triangles {a,b,c} and {d,e,f} joined by a single bridge edge a-d.
    fn barbell() -> Graph {
        let mut g = Graph::new();
        for id in ["a", "b", "c", "d", "e", "f"] {
            g.add_node(node(id));
        }
        // Triangle 1
        for (s, t) in [
            ("a", "b"),
            ("b", "a"),
            ("b", "c"),
            ("c", "b"),
            ("a", "c"),
            ("c", "a"),
        ] {
            g.add_edge(edge(s, t));
        }
        // Triangle 2
        for (s, t) in [
            ("d", "e"),
            ("e", "d"),
            ("e", "f"),
            ("f", "e"),
            ("d", "f"),
            ("f", "d"),
        ] {
            g.add_edge(edge(s, t));
        }
        // Bridge
        g.add_edge(edge("a", "d"));
        g.add_edge(edge("d", "a"));
        g
    }

    // ── Known-spectrum correctness ───────────────────────────────────────────

    #[test]
    fn k2_algebraic_connectivity_is_2() {
        // K₂ normalized Laplacian eigenvalues: {0, 2}. The Fiedler value must be ≈ 2.
        let ac = algebraic_connectivity(&k2()).unwrap();
        assert!(
            (ac - 2.0).abs() < 1e-6,
            "K₂ Fiedler value should be 2.0, got {ac}"
        );
    }

    #[test]
    fn two_disconnected_triangles_are_detected() {
        // Disconnected graph → algebraic connectivity ≈ 0.
        let ac = algebraic_connectivity(&two_triangles()).unwrap();
        assert!(
            ac.abs() < 1e-6,
            "Disconnected graph should have Fiedler ≈ 0, got {ac}"
        );

        // Fiedler vector must have the same sign within each triangle and opposite sign across.
        let fv = fiedler_vector(&two_triangles()).unwrap();
        let by_id: HashMap<&str, f64> = fv.iter().map(|(id, v)| (id.0.as_str(), *v)).collect();

        // All of {a,b,c} should have the same sign, all of {d,e,f} the other sign.
        let sign_a = by_id["a"].signum();
        for id in ["a", "b", "c"] {
            let s = by_id[id].signum();
            assert_eq!(
                s, sign_a,
                "{id} should be in the same component as 'a' (sign {sign_a}), got {s}"
            );
        }
        let sign_d = by_id["d"].signum();
        assert_ne!(
            sign_a, sign_d,
            "The two triangles should be on opposite sides of the Fiedler cut"
        );
        for id in ["d", "e", "f"] {
            let s = by_id[id].signum();
            assert_eq!(
                s, sign_d,
                "{id} should be in the same component as 'd' (sign {sign_d}), got {s}"
            );
        }
    }

    #[test]
    fn barbell_fiedler_splits_the_two_clusters() {
        // Barbell: two triangles joined by a bridge. The Fiedler vector should
        // assign positive coordinates to one triangle and negative to the other.
        let fv = fiedler_vector(&barbell()).unwrap();
        let by_id: HashMap<&str, f64> = fv.iter().map(|(id, v)| (id.0.as_str(), *v)).collect();

        // The Fiedler vector sign should separate {a,b,c} from {d,e,f}.
        // After sign canonicalization, we don't know which side is positive, so
        // just verify the two groups have opposite signs.
        let sign_a = by_id["a"].signum();
        for id in ["a", "b", "c"] {
            assert_eq!(
                by_id[id].signum(),
                sign_a,
                "{id} should be in the same cluster as 'a'"
            );
        }
        let sign_d = by_id["d"].signum();
        assert_ne!(
            sign_a, sign_d,
            "The two halves of the barbell should be on opposite sides of the Fiedler cut"
        );
        for id in ["d", "e", "f"] {
            assert_eq!(
                by_id[id].signum(),
                sign_d,
                "{id} should be in the same cluster as 'd'"
            );
        }
    }

    // ── spectral_embedding shape + ordering ──────────────────────────────────

    #[test]
    fn embedding_shape_and_nondecreasing_eigenvalues() {
        let g = barbell();
        let n = g.node_count(); // 6
        let k = 3;
        let emb = spectral_embedding(&g, k).unwrap();

        assert_eq!(emb.nodes.len(), n, "node list length");
        assert_eq!(emb.eigenvalues.len(), k, "eigenvalue count");
        assert_eq!(emb.coords.len(), n, "coords row count");
        for c in &emb.coords {
            assert_eq!(c.len(), k, "each coord vector has length k");
        }
        // Eigenvalues must be non-decreasing.
        for pair in emb.eigenvalues.windows(2) {
            assert!(
                pair[0] <= pair[1] + 1e-12,
                "eigenvalues not non-decreasing: {:?}",
                emb.eigenvalues
            );
        }
    }

    // ── Determinism ──────────────────────────────────────────────────────────

    #[test]
    fn fiedler_vector_is_deterministic_across_insertion_orders() {
        // Build the same barbell in two different insertion orders; the Fiedler vectors
        // must be identical after sign canonicalization (within 1e-9).
        let g1 = barbell();

        // Build in reverse order.
        let mut g2 = Graph::new();
        for id in ["f", "e", "d", "c", "b", "a"] {
            g2.add_node(node(id));
        }
        // Triangle 2 first
        for (s, t) in [
            ("d", "f"),
            ("f", "d"),
            ("e", "f"),
            ("f", "e"),
            ("d", "e"),
            ("e", "d"),
        ] {
            g2.add_edge(edge(s, t));
        }
        // Triangle 1
        for (s, t) in [
            ("a", "c"),
            ("c", "a"),
            ("b", "c"),
            ("c", "b"),
            ("a", "b"),
            ("b", "a"),
        ] {
            g2.add_edge(edge(s, t));
        }
        // Bridge
        g2.add_edge(edge("d", "a"));
        g2.add_edge(edge("a", "d"));

        let fv1: HashMap<String, f64> = fiedler_vector(&g1)
            .unwrap()
            .into_iter()
            .map(|(id, v)| (id.0, v))
            .collect();
        let fv2: HashMap<String, f64> = fiedler_vector(&g2)
            .unwrap()
            .into_iter()
            .map(|(id, v)| (id.0, v))
            .collect();

        for (id, v1) in &fv1 {
            let v2 = fv2[id];
            assert!(
                (v1 - v2).abs() < 1e-9,
                "node {id}: g1={v1} vs g2={v2} (insertion order should not matter)"
            );
        }
    }

    // ── Error cases ──────────────────────────────────────────────────────────

    #[test]
    fn empty_graph_returns_empty_graph_error() {
        let g = Graph::new();
        assert_eq!(algebraic_connectivity(&g), Err(SpectralError::EmptyGraph));
        assert_eq!(fiedler_vector(&g), Err(SpectralError::EmptyGraph));
        assert_eq!(spectral_embedding(&g, 1), Err(SpectralError::EmptyGraph));
    }

    #[test]
    fn single_node_graph_returns_too_few_nodes() {
        let mut g = Graph::new();
        g.add_node(node("solo"));
        assert_eq!(
            fiedler_vector(&g),
            Err(SpectralError::TooFewNodes { have: 1, need: 2 })
        );
        assert_eq!(
            spectral_embedding(&g, 1),
            Err(SpectralError::TooFewNodes { have: 1, need: 2 })
        );
        // algebraic_connectivity also needs 2
        assert_eq!(
            algebraic_connectivity(&g),
            Err(SpectralError::TooFewNodes { have: 1, need: 2 })
        );
    }

    #[test]
    fn too_few_nodes_for_embedding_k() {
        // 3-node graph, asking for k=3 non-trivial eigenvectors → need 4 nodes.
        let mut g = Graph::new();
        for id in ["x", "y", "z"] {
            g.add_node(node(id));
        }
        g.add_edge(edge("x", "y"));
        g.add_edge(edge("y", "x"));
        assert_eq!(
            spectral_embedding(&g, 3),
            Err(SpectralError::TooFewNodes { have: 3, need: 4 })
        );
    }
}
