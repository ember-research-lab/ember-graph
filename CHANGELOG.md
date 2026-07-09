# Changelog

All notable changes to ember-graph are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — verification substrate + tensor / crypto / spectral layers

The v0.2 line turns the graph from a retrieval structure into a **verification
substrate**: an orchestrator can pull the signed fact behind a claim in O(1) and
check it against ground truth rather than a lossy summary. Core stays
zero-dependency; `crypto` and `spectral` are opt-in features.

### Added
- **Stable u64 tensor-address index** (`NodeIndex`): content-id → dense, monotonic
  `u64`, never reused; tombstones on delete, lineage on merge — keeps GNN/tensor
  consumers row-aligned across mutations.
- **Crypto-agile provenance envelope** (`Provenance`, `VerifierRegistry`): every
  extracted node/edge carries a tamper-evident, offline-verifiable envelope
  (asserting agent, basis, content-addressed source). Algorithm is named, never
  hardcoded, so Ed25519 → ML-DSA is additive.
- **`ember-crypto` provenance adapter** (`crypto` feature): `EmberSigner` /
  `EmberVerifier` + `TrustStore` implementing the platform/tenant/agent trust
  model (see `design/trust-model.md`).
- **O(1) verification retrieval** (`ProvenanceIndex`, `Graph::fact_view`): fetch a
  node, its edges, its confidence, and its signed provenance in one hop.
- **Self-contained artifact format** (`Artifact`): one versioned, deterministic,
  strictly-parsed byte container for graph + index + provenance, with a JSON debug
  export and corruption rejection.
- **Attribute propagation** (`Graph::propagate`): generic meet-semilattice fixpoint
  over edges (buckets → trust mapping; bounded by node count).
- **Spectral layer** (`spectral` feature, `faer`): normalized-Laplacian
  `algebraic_connectivity`, `fiedler_vector`, `spectral_embedding(k)`.
- **Erasure** (`Graph::forget`): node + incident-edge removal for GDPR-style
  deletion.
- **Whole-graph iteration** (`Graph::node_ids`).

### Notes
- Core remains zero-dependency with `#![forbid(unsafe_code)]`.
- fmt + clippy `-D warnings` clean across default / `crypto` / `spectral`; 46 tests.

## [0.1.0] — engine core

### Added
- Typed node/edge model with content-derived identity (`NodeId::canonical`) as the
  cross-source join key.
- Discrete confidence rubric (`Confidence`: EXTRACTED / INFERRED(tier) / AMBIGUOUS)
  with bucketed `score()`.
- Token-budgeted BFS query primitive (`Graph::query`) with lexical seeding and
  hub-avoidance, plus `neighbors` and `shortest_path`.

[0.2.0]: https://github.com/ember-research-lab/ember-graph/releases/tag/v0.2.0
[0.1.0]: https://github.com/ember-research-lab/ember-graph/releases/tag/v0.1.0
