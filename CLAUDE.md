# ember-graph — zero-dependency knowledge-graph engine (multi-consumer Ember capability)

Single crate: typed node/edge model, content-derived `NodeId`s as the join key, discrete confidence
rubric, token-budgeted BFS query. Core is zero-dep; `crypto` (ember-crypto) and `spectral` (faer)
are opt-in features. Read `README.md` and `design/trust-model.md` first.

- **Two consumers pin it differently:**
  - claude-cortex `cortex-graph` → git dep, `tag = "v0.2.0"` (root `Cargo.toml`).
  - ember-smb-platform `adapter-graph` → path dep `../../../ember-graph` (tracks this checkout).
  ⇒ A breaking API change needs a new tag (bump `Cargo.toml` `version` + `CHANGELOG.md`) **and**
  both consumers updated — the path consumer breaks immediately, the tag consumer only on re-pin.
- **Test:** `cargo test` (also `cargo test --features spectral` / `--features crypto` when touching
  those layers). No CI workflow in this repo — the local run is the bar.
