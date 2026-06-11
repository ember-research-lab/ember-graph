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

Status: v0.1 — the engine core (types + store + query). Zero-dep, `#![forbid(unsafe_code)]`,
tested. Next: a thin extraction-contract helper (the JSON schema + injection-hardened source
wrapping) and per-consumer adapters.
