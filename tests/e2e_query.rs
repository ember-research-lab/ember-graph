//! End-to-end release-candidate test.
//!
//! Exercises the exact capability cortex's v-next retrieval layer depends on:
//! project ledger-style learning nodes into a graph, then answer a question
//! with a compact subgraph that is (a) *relevant* — related learnings reached by
//! traversal, unrelated ones scoped out — and (b) *budget-bounded*. Verified
//! both in-memory and after a full artifact serialize/deserialize round-trip.

use ember_graph::{Artifact, Confidence, Edge, Graph, InferredTier, Node, NodeId, NodeIndex};

fn node(ns: &str, ent: &str, label: &str, kind: &str) -> Node {
    Node {
        id: NodeId::canonical(ns, ent),
        label: label.to_string(),
        kind: kind.to_string(),
        source: None,
    }
}

fn edge(src: NodeId, dst: NodeId, rel: &str, c: Confidence) -> Edge {
    Edge {
        source: src,
        target: dst,
        relation: rel.to_string(),
        confidence: c,
    }
}

/// A tiny cortex-ledger projection: one topic node whose label lexically matches
/// the query, three learnings reachable only by TRAVERSAL (their labels do not
/// contain the query terms), and one unrelated learning that must be scoped out.
fn ledger_projection() -> Graph {
    let mut g = Graph::new();
    g.add_node(node(
        "topic",
        "confidence-model",
        "confidence model",
        "topic",
    ));
    g.add_node(node(
        "learning",
        "decay",
        "decay relaxes toward the prior on disuse",
        "learning",
    ));
    g.add_node(node(
        "learning",
        "corroboration",
        "corroboration counts directly",
        "learning",
    ));
    g.add_node(node(
        "learning",
        "reclassification",
        "Validated and Contested reclassification",
        "learning",
    ));
    // Unrelated learning — must NOT surface for a confidence query.
    g.add_node(node(
        "learning",
        "wal",
        "WAL bloat corrupts the whale signal database",
        "learning",
    ));

    let t = NodeId::canonical("topic", "confidence-model");
    let l_decay = NodeId::canonical("learning", "decay");
    let l_corr = NodeId::canonical("learning", "corroboration");
    let l_recl = NodeId::canonical("learning", "reclassification");

    g.add_edge(edge(
        t.clone(),
        l_decay.clone(),
        "includes",
        Confidence::Extracted,
    ));
    g.add_edge(edge(t, l_corr, "includes", Confidence::Extracted));
    // Reached only at depth 2, via the decay learning — proves multi-hop recall.
    g.add_edge(edge(
        l_decay,
        l_recl,
        "leads_to",
        Confidence::Inferred(InferredTier::Clear),
    ));
    g
}

#[test]
fn query_returns_relevant_budget_bounded_subgraph() {
    let g = ledger_projection();
    let budget = 4000;
    let out = g.query("confidence model", 2, budget);

    assert!(!out.is_empty(), "query returned nothing");
    assert!(
        out.len() <= budget,
        "result {} exceeded budget {budget}",
        out.len()
    );
    // Reached by 1-hop and 2-hop traversal, NOT by lexical match on the query.
    assert!(
        out.contains("corroboration"),
        "1-hop related learning missing:\n{out}"
    );
    assert!(
        out.contains("Validated"),
        "2-hop related learning missing:\n{out}"
    );
    // Unrelated learning must be scoped out.
    assert!(
        !out.contains("whale signal"),
        "irrelevant learning leaked:\n{out}"
    );
}

#[test]
fn direct_seed_query_finds_learning() {
    let g = ledger_projection();
    let out = g.query("corroboration", 1, 2000);
    assert!(
        out.contains("corroboration"),
        "direct-seed query failed:\n{out}"
    );
}

#[test]
fn query_survives_artifact_round_trip() {
    let g = ledger_projection();
    let bytes = Artifact::new(g, NodeIndex::new()).to_bytes();
    let restored = Artifact::from_bytes(&bytes).expect("round-trip parse");

    let out = restored.graph.query("confidence model", 2, 4000);
    assert!(
        out.contains("corroboration"),
        "query broke after round-trip:\n{out}"
    );
    assert!(
        !out.contains("whale signal"),
        "scoping broke after round-trip:\n{out}"
    );
}
