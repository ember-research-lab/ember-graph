# ember-graph

A reusable, **zero-dependency** knowledge-graph engine for the Ember stack — a clean-room
reimplementation of the patterns proven by [Graphify](https://github.com/safishamsi/graphify)
(MIT), built as a *multi-consumer Ember capability* rather than a coding-assistant tool.

## Why

Across Ember we keep needing the same thing: turn a pile of entities + documents into a graph an
agent can **reason over by traversal** instead of re-reading everything. Rather than vendor a
heavy, single-tenant dev tool, this is the small, dependency-free core of that idea — reusable by
the SMB platform (a business knowledge graph), the research repos (manuscript ↔ code ↔ Lean
lineage), and agent context generally.

## What's here (the three load-bearing ideas)

1. **Content-derived node ids as the join key** (`NodeId::canonical`) — the same entity from two
   sources collides onto one node, so a structured record and an LLM-extracted mention *meet*
   without an entity-resolution service.
2. **A discrete confidence rubric** (`Confidence` / `Confidence::score`) — every edge is
   EXTRACTED / INFERRED(tier) / AMBIGUOUS with a *bucketed* score (never a wishy-washy 0.5). A
   calibrated, ledgerable trust signal for free.
3. **A token-budgeted BFS query** (`Graph::query`) — seed on the best-matching nodes, walk the
   neighbourhood with **hub-avoidance**, render a compact subgraph within a budget. No vector DB,
   no LLM in the hot path. Plus `neighbors`, `shortest_path`.

## v0.2 — verification substrate + tensor / crypto / spectral layers

The graph's deepest role isn't retrieval, it's **verification**: an orchestrator works from cheap
summaries that drop nuance it can't see, so it verifies a claim by pulling the **signed fact behind
it** from the graph in O(1) and checking against that ground truth — not against the summary it came
from. The v0.2 layers make that real:

- **Stable u64 tensor-address index** (`NodeIndex`) — content-id → dense, monotonic `u64` (never
  reused; tombstones on delete, lineage on merge) so a GNN/tensor consumer keeps aligned rows.
- **Crypto-agile provenance envelope** (`Provenance`, `VerifierRegistry`) — every extracted
  node/edge carries a tamper-evident, offline-verifiable envelope (who asserted it, on what basis,
  over what source). The algorithm is **named, never hardcoded**, so Ed25519 → ML-DSA is additive.
- **Real signing via `ember-crypto`** (`crypto` feature) — `EmberSigner`/`EmberVerifier` +
  `TrustStore` implement the platform/tenant/agent **trust model** (`design/trust-model.md`):
  delegation chain for authority, per-action signature for integrity, all offline.
- **O(1) verification retrieval** (`ProvenanceIndex`, `Graph::fact_view`) — fetch the signed
  fact + its confidence + provenance for a subject in one hop.
- **Self-contained artifact format** (`Artifact`) — one versioned, deterministic, strictly-parsed
  byte container for the graph + index + signed provenance (+ a JSON debug export).
- **Spectral layer** (`spectral` feature, `faer`) — normalized-Laplacian `algebraic_connectivity`,
  `fiedler_vector`, `spectral_embedding(k)` for clustering / community structure.

The **core stays zero-dependency**; `crypto` and `spectral` are opt-in features.

## What's *not* here (by design)

**Extraction** — turning documents into nodes/edges via an LLM — is the *consumer's* job (it owns
the model + the prompt; ember-graph is the schema it targets). The structured side (records →
nodes/edges) is a pure projection the consumer feeds in directly. ember-graph is the **graph +
query**, not an LLM client or a JSON parser.

## Use

```rust
use ember_graph::{Graph, Node, Edge, NodeId, Confidence, InferredTier};

let mut g = Graph::new();
// structured projection (records → nodes/edges, high confidence)
g.add_edge(Edge { source: NodeId::canonical("t7","John Doe"),
                  target: NodeId::canonical("t7","Acme LLC"),
                  relation: "owns".into(), confidence: Confidence::Extracted });
// the agent asks a multi-hop question; gets a compact subgraph, not the whole corpus
let context = g.query("which account does John Doe's email concern", 3, 2000);
```

Status: **v0.2 complete** — engine core (types + store + query) + the verification substrate
(u64 index, crypto-agile provenance + `ember-crypto` adapter + trust model, O(1) `fact_view`,
self-contained artifact format) + the spectral layer. Core zero-dep, `#![forbid(unsafe_code)]`,
fully tested (clippy `-D warnings` + fmt clean across default / `crypto` / `spectral`). Next: the
SMB B3 wiring (business knowledge-graph backend) and a thin extraction-contract helper. See
`design/trust-model.md` for the signing model.
