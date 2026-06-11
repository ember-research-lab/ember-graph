//! # ember-graph
//!
//! A reusable, **zero-dependency** knowledge-graph engine for the Ember stack — a clean-room
//! reimplementation of the patterns proven by [Graphify](https://github.com/safishamsi/graphify)
//! (MIT), built as a multi-consumer Ember capability rather than a coding-assistant tool.
//!
//! Three ideas carry the value, and they're all here, dependency-free:
//!
//! 1. **A typed node/edge model with a content-derived id as the join key** — the same entity
//!    yields the same [`NodeId`] across sources, so a structured record and an LLM-extracted mention
//!    *meet at the same node* without an entity-resolution service.
//! 2. **A discrete confidence rubric** ([`Confidence`] + [`Confidence::score`]) — every edge is
//!    labelled EXTRACTED / INFERRED / AMBIGUOUS with a *bucketed* numeric score (models collapse a
//!    continuous range to a bimodal blob, so the buckets are deliberate). This gives a consumer a
//!    calibrated, ledgerable trust signal "for free."
//! 3. **A token-budgeted BFS query** ([`Graph::query`]) — answer a question by seeding on the
//!    best-matching nodes and walking their neighbourhood with **hub-avoidance**, rendering a
//!    compact subgraph within a token budget. No vector DB, no LLM call in the hot path — the agent
//!    reasons over a small map instead of re-reading the corpus.
//!
//! **Extraction** (turning documents into nodes/edges via an LLM) is the *consumer's* job — it owns
//! the model and the prompt; ember-graph is the schema it targets + the store + the query. The
//! structured side (records → nodes/edges) is a pure projection the consumer feeds in directly.

#![forbid(unsafe_code)]

pub mod index;
pub use index::NodeIndex;

use std::collections::{HashMap, HashSet, VecDeque};

/// A node id — **deterministic and content-derived**, so the same entity from two sources collides
/// onto one node (the join key). Build it with [`NodeId::canonical`] for a normalized id, or wrap an
/// already-canonical string.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub String);

impl NodeId {
    /// Canonicalize `(namespace, entity)` into a stable id: lowercased, non-alphanumerics → `_`,
    /// collapsed. `("tenant7", "Acme LLC")` and `("tenant7", "acme  llc")` both → `tenant7::acme_llc`.
    /// This is the dedup/join key — a records projection and a doc extractor that produce the same
    /// `(namespace, entity)` land on the same node.
    pub fn canonical(namespace: &str, entity: &str) -> Self {
        let norm = |s: &str| {
            let mut out = String::with_capacity(s.len());
            let mut prev_us = false;
            for c in s.chars() {
                if c.is_ascii_alphanumeric() {
                    out.extend(c.to_lowercase());
                    prev_us = false;
                } else if !prev_us {
                    out.push('_');
                    prev_us = true;
                }
            }
            out.trim_matches('_').to_string()
        };
        NodeId(format!("{}::{}", norm(namespace), norm(entity)))
    }
}

/// How much to trust an edge — categorical, mirroring Graphify's rubric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Confidence {
    /// Read directly from structured/authoritative data (a foreign key, a records projection). 1.0.
    Extracted,
    /// An LLM/heuristic inferred it. Bucketed (see [`Confidence::score`]) — never a wishy-washy 0.5.
    Inferred(InferredTier),
    /// The source is unclear; surfaced but low-trust.
    Ambiguous,
}

/// Discrete tiers for an inferred edge. Buckets, not a continuum — production telemetry shows models
/// collapse a free range to a bimodal 0.5/0.85, so the rubric forces a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferredTier {
    /// Strongly implied by explicit local context. 0.95.
    Strong,
    /// Clear contextual link. 0.85.
    Clear,
    /// Reasonable. 0.75.
    Reasonable,
    /// Weak but more than a guess. 0.65.
    Weak,
    /// Tenuous. 0.55.
    Tenuous,
}

impl Confidence {
    /// The bucketed numeric score (Graphify's rubric: 1.0 extracted; 0.95/0.85/0.75/0.65/0.55
    /// inferred tiers; 0.2 ambiguous). 0.5 is deliberately unreachable.
    pub fn score(self) -> f32 {
        match self {
            Confidence::Extracted => 1.0,
            Confidence::Inferred(InferredTier::Strong) => 0.95,
            Confidence::Inferred(InferredTier::Clear) => 0.85,
            Confidence::Inferred(InferredTier::Reasonable) => 0.75,
            Confidence::Inferred(InferredTier::Weak) => 0.65,
            Confidence::Inferred(InferredTier::Tenuous) => 0.55,
            Confidence::Ambiguous => 0.2,
        }
    }
}

/// A graph node — an entity (a customer, a product, a document, a concept).
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    /// Human-facing label (what the query matches against), e.g. "Acme LLC".
    pub label: String,
    /// What kind of thing it is, consumer-defined: "contact" | "product" | "document" | …
    pub kind: String,
    /// Where it came from (a record id, a source-file path + sha, …) — for provenance.
    pub source: Option<String>,
}

/// A directed, typed, confidence-weighted edge. By convention **source is the actor, target the
/// acted-upon** (so direction is stable across extractors).
#[derive(Clone, Debug)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    /// The relationship, consumer-defined: "placed_by" | "bills" | "mentions" | "same_as" | …
    pub relation: String,
    pub confidence: Confidence,
}

/// The knowledge graph: nodes + directed adjacency. Incremental and **never silently shrinks** (a
/// merge only adds / upgrades). Zero-dep; an in-memory adjacency structure.
#[derive(Default)]
pub struct Graph {
    nodes: HashMap<NodeId, Node>,
    /// source -> outgoing edges.
    out: HashMap<NodeId, Vec<Edge>>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or upgrade a node. Idempotent on id; a node with a `source` (more authoritative) overwrites
    /// a source-less one, and a non-empty kind/label fills a blank — but it never deletes.
    pub fn add_node(&mut self, node: Node) {
        match self.nodes.get_mut(&node.id) {
            Some(existing) => {
                if existing.source.is_none() && node.source.is_some() {
                    existing.source = node.source;
                }
                if existing.label.is_empty() {
                    existing.label = node.label;
                }
                if existing.kind.is_empty() {
                    existing.kind = node.kind;
                }
            }
            None => {
                self.nodes.insert(node.id.clone(), node);
            }
        }
    }

    /// Add an edge. Both endpoints are auto-created as bare nodes if absent (so an extractor can emit
    /// an edge to a record id it didn't also emit a node for — the records projection fills it later).
    /// A duplicate (same source/target/relation) keeps the **higher** confidence.
    pub fn add_edge(&mut self, edge: Edge) {
        for id in [&edge.source, &edge.target] {
            self.nodes.entry(id.clone()).or_insert_with(|| Node {
                id: id.clone(),
                label: id.0.clone(),
                kind: String::new(),
                source: None,
            });
        }
        let list = self.out.entry(edge.source.clone()).or_default();
        if let Some(e) = list
            .iter_mut()
            .find(|e| e.target == edge.target && e.relation == edge.relation)
        {
            if edge.confidence.score() > e.confidence.score() {
                e.confidence = edge.confidence;
            }
        } else {
            list.push(edge);
        }
    }

    /// Merge another graph in (incremental; never shrinks).
    pub fn merge(&mut self, other: Graph) {
        for n in other.nodes.into_values() {
            self.add_node(n);
        }
        for edges in other.out.into_values() {
            for e in edges {
                self.add_edge(e);
            }
        }
    }

    pub fn node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Outgoing edges from a node.
    pub fn neighbors(&self, id: &NodeId) -> &[Edge] {
        self.out.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Undirected degree of a node (in + out) — used for hub-avoidance.
    fn degree(&self, id: &NodeId) -> usize {
        let outd = self.out.get(id).map(|v| v.len()).unwrap_or(0);
        let ind = self
            .out
            .values()
            .flatten()
            .filter(|e| &e.target == id)
            .count();
        outd + ind
    }

    /// The p99 degree — the hub threshold (nodes above this are "god nodes" we don't expand through,
    /// so a query frontier can't explode).
    fn hub_threshold(&self) -> usize {
        let mut degs: Vec<usize> = self.nodes.keys().map(|id| self.degree(id)).collect();
        if degs.is_empty() {
            return usize::MAX;
        }
        degs.sort_unstable();
        let idx = ((degs.len() as f64) * 0.99).floor() as usize;
        degs[idx.min(degs.len() - 1)].max(1)
    }

    /// Shortest directed path between two nodes (BFS), or `None` if unreachable.
    pub fn shortest_path(&self, from: &NodeId, to: &NodeId) -> Option<Vec<NodeId>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return None;
        }
        let mut prev: HashMap<NodeId, NodeId> = HashMap::new();
        let mut seen: HashSet<NodeId> = HashSet::from([from.clone()]);
        let mut q = VecDeque::from([from.clone()]);
        while let Some(cur) = q.pop_front() {
            if &cur == to {
                let mut path = vec![to.clone()];
                let mut c = to;
                while let Some(p) = prev.get(c) {
                    path.push(p.clone());
                    c = p;
                }
                path.reverse();
                return Some(path);
            }
            for e in self.neighbors(&cur) {
                if seen.insert(e.target.clone()) {
                    prev.insert(e.target.clone(), cur.clone());
                    q.push_back(e.target.clone());
                }
            }
        }
        None
    }

    /// **The query primitive.** Answer `question` by (1) lexically scoring nodes by label overlap
    /// (exact > prefix > substring, IDF-ish so rare terms count more), (2) seeding on the best few,
    /// (3) BFS to `depth` with **hub-avoidance** (don't expand through high-degree nodes), (4)
    /// rendering the reached subgraph as text within `budget_chars`. No LLM, no vectors — a compact
    /// neighbourhood the caller feeds to an agent instead of the whole corpus.
    pub fn query(&self, question: &str, depth: usize, budget_chars: usize) -> String {
        let seeds = self.seed(question, 3);
        if seeds.is_empty() {
            return String::new();
        }
        let hub = self.hub_threshold();
        let mut reached: Vec<NodeId> = Vec::new();
        let mut seen: HashSet<NodeId> = HashSet::new();
        let mut frontier: VecDeque<(NodeId, usize)> =
            seeds.iter().cloned().map(|s| (s, 0usize)).collect();
        for s in &seeds {
            seen.insert(s.clone());
        }
        while let Some((cur, d)) = frontier.pop_front() {
            reached.push(cur.clone());
            // Don't expand through a hub (it would flood the frontier), but the hub itself is kept.
            if d >= depth || self.degree(&cur) > hub {
                continue;
            }
            for e in self.neighbors(&cur) {
                if seen.insert(e.target.clone()) {
                    frontier.push_back((e.target.clone(), d + 1));
                }
            }
        }
        self.render(&reached, budget_chars)
    }

    /// Pick up to `k` seed nodes whose labels best match the question's terms.
    fn seed(&self, question: &str, k: usize) -> Vec<NodeId> {
        let terms = tokenize(question);
        if terms.is_empty() {
            return Vec::new();
        }
        // crude IDF: a term in few node labels is worth more.
        let total = self.nodes.len().max(1) as f32;
        let df = |t: &str| -> f32 {
            let n = self
                .nodes
                .values()
                .filter(|nd| nd.label.to_lowercase().contains(t))
                .count()
                .max(1) as f32;
            (total / n).ln().max(0.0) + 1.0
        };
        let mut scored: Vec<(f32, NodeId)> = self
            .nodes
            .values()
            .map(|nd| {
                let label = nd.label.to_lowercase();
                let mut s = 0.0;
                for t in &terms {
                    if label == *t {
                        s += 3.0 * df(t);
                    } else if label.starts_with(t.as_str()) {
                        s += 2.0 * df(t);
                    } else if label.contains(t.as_str()) {
                        s += 1.0 * df(t);
                    }
                }
                (s, nd.id.clone())
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // score-gap cutoff: drop seeds far below the best (noise terms shouldn't seed).
        let best = scored.first().map(|(s, _)| *s).unwrap_or(0.0);
        scored
            .into_iter()
            .take(k)
            .filter(|(s, _)| *s >= best * 0.25)
            .map(|(_, id)| id)
            .collect()
    }

    /// Render a subgraph (the reached nodes + the edges among them) as compact text within a budget.
    fn render(&self, reached: &[NodeId], budget_chars: usize) -> String {
        let set: HashSet<&NodeId> = reached.iter().collect();
        let mut out = String::new();
        for id in reached {
            if out.len() >= budget_chars {
                out.push_str("…\n");
                break;
            }
            let Some(nd) = self.nodes.get(id) else {
                continue;
            };
            let kind = if nd.kind.is_empty() { "node" } else { &nd.kind };
            out.push_str(&format!("[{}] {}\n", kind, nd.label));
            for e in self.neighbors(id) {
                if set.contains(&e.target) {
                    if out.len() >= budget_chars {
                        break;
                    }
                    let tgt = self
                        .nodes
                        .get(&e.target)
                        .map(|n| n.label.as_str())
                        .unwrap_or(&e.target.0);
                    out.push_str(&format!(
                        "  → {} {} ({:.2})\n",
                        e.relation,
                        tgt,
                        e.confidence.score()
                    ));
                }
            }
        }
        out
    }
}

/// Lowercase alphanumeric word tokens.
fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: &str, label: &str, kind: &str) -> Node {
        Node {
            id: NodeId(id.into()),
            label: label.into(),
            kind: kind.into(),
            source: None,
        }
    }
    fn e(s: &str, t: &str, rel: &str, c: Confidence) -> Edge {
        Edge {
            source: NodeId(s.into()),
            target: NodeId(t.into()),
            relation: rel.into(),
            confidence: c,
        }
    }

    #[test]
    fn canonical_id_is_the_join_key() {
        assert_eq!(
            NodeId::canonical("tenant7", "Acme LLC"),
            NodeId::canonical("tenant7", "acme  llc")
        );
        assert_eq!(NodeId::canonical("t", "John Doe").0, "t::john_doe");
    }

    #[test]
    fn confidence_rubric_buckets_never_05() {
        assert_eq!(Confidence::Extracted.score(), 1.0);
        assert_eq!(Confidence::Inferred(InferredTier::Clear).score(), 0.85);
        assert_eq!(Confidence::Ambiguous.score(), 0.2);
        for t in [
            InferredTier::Strong,
            InferredTier::Clear,
            InferredTier::Reasonable,
            InferredTier::Weak,
            InferredTier::Tenuous,
        ] {
            assert!((Confidence::Inferred(t).score() - 0.5).abs() > 0.01);
        }
    }

    #[test]
    fn add_node_upgrades_never_shrinks() {
        let mut g = Graph::new();
        g.add_node(n("a", "Acme", ""));
        g.add_node(Node {
            id: NodeId("a".into()),
            label: "Acme".into(),
            kind: "contact".into(),
            source: Some("rec:1".into()),
        });
        let got = g.node(&NodeId("a".into())).unwrap();
        assert_eq!(got.kind, "contact"); // blank kind filled
        assert_eq!(got.source.as_deref(), Some("rec:1")); // source-less upgraded
        assert_eq!(g.node_count(), 1); // not duplicated
    }

    #[test]
    fn add_edge_keeps_higher_confidence_and_autocreates_endpoints() {
        let mut g = Graph::new();
        g.add_edge(e("a", "b", "mentions", Confidence::Ambiguous));
        g.add_edge(e("a", "b", "mentions", Confidence::Extracted)); // upgrade
        assert_eq!(g.neighbors(&NodeId("a".into())).len(), 1);
        assert_eq!(g.neighbors(&NodeId("a".into()))[0].confidence.score(), 1.0);
        assert!(g.node(&NodeId("b".into())).is_some()); // auto-created
    }

    #[test]
    fn query_does_a_multi_hop_join() {
        // The email-disambiguation case: John Doe → Acme LLC → an unpaid invoice.
        let mut g = Graph::new();
        g.add_node(n("c:john", "John Doe", "contact"));
        g.add_node(n("b:acme", "Acme LLC", "business"));
        g.add_node(n("i:42", "Invoice 42 (unpaid)", "invoice"));
        g.add_node(n("p:rug", "9x12 wool rug", "product"));
        g.add_edge(e("c:john", "b:acme", "owns", Confidence::Extracted));
        g.add_edge(e("b:acme", "i:42", "owes", Confidence::Extracted));
        g.add_edge(e(
            "c:john",
            "p:rug",
            "asked_about",
            Confidence::Inferred(InferredTier::Clear),
        ));

        let out = g.query("which account does John Doe's email concern", 3, 2000);
        assert!(out.contains("John Doe"), "seeds on John: {out}");
        assert!(out.contains("Acme LLC"), "reaches his business: {out}");
        assert!(
            out.contains("Invoice 42"),
            "reaches the multi-hop invoice: {out}"
        );

        // shortest path John → invoice is 2 hops via Acme.
        let path = g
            .shortest_path(&NodeId("c:john".into()), &NodeId("i:42".into()))
            .unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn query_budget_truncates() {
        let mut g = Graph::new();
        for i in 0..50 {
            g.add_node(n(&format!("n{i}"), &format!("widget {i}"), "thing"));
            g.add_edge(e("n0", &format!("n{i}"), "near", Confidence::Extracted));
        }
        let out = g.query("widget", 3, 120);
        assert!(
            out.len() <= 140,
            "respects the char budget: {} chars",
            out.len()
        );
    }

    #[test]
    fn empty_or_unmatched_query_is_empty() {
        let mut g = Graph::new();
        g.add_node(n("a", "Acme", "contact"));
        assert!(g.query("zzzz nonexistent", 2, 500).is_empty());
        assert!(g.query("", 2, 500).is_empty());
    }
}
