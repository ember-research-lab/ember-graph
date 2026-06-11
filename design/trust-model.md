# ember-graph trust model — the three-way signing-key decision

**Status:** decided (2026-06-11). This resolves the "open dependency" flagged in `src/provenance.rs`
(the SMB platform's platform / tenant / agent trust model). **The envelope schema does not change** —
`Provenance { asserting_agent, basis, parent_hash, algorithm, attestations: Vec<Attestation> }` is
final. What this doc decides is *what the attestations mean* and *how a verifier establishes trust*.

## The three legs

A poisoned edge from a compromised agent is a poisoned fact every downstream agent reasons from
(`provenance.rs` threat model: *a compromised agent is a compromised insider*). Three distinct
authorities bound that risk, and they are **not** the same key:

1. **Platform attests runtime** — the platform vouches that an agent runtime is the vetted binary in
   the right sandbox. This is the **trust anchor**: a single platform **root** key.
2. **Tenant scopes namespace** — a tenant owns a data namespace (`tenant7::…` in `NodeId::canonical`).
   Cross-namespace assertions are invalid. A per-tenant **root** key.
3. **Agent signs action** — each running agent is a principal with its own signing identity and signs
   every fact it asserts. **This is the signature already in the ember-graph envelope** (the
   `agent-signs-action` attestation over `signed_payload(subject_id)`).

## The model: delegation chain for *authority*, per-action signature for *integrity*

Two separable properties, two mechanisms — the standard, well-understood split (don't invent crypto):

- **Integrity** (was this exact fact signed by this agent, untampered?) → the **agent's signature over
  `signed_payload(subject_id)`**, carried in the envelope. This is exactly what `Provenance::verify`
  checks today. A flipped byte anywhere fails it; a signature can't be lifted to another subject
  (subject id is in the payload).
- **Authority** (was that agent *allowed* to assert this, for this tenant/namespace?) → a **delegation
  chain**, PKI-style:
  - platform **root** → signs each tenant root key → *"tenant T owns namespace N"*
  - tenant **root** → signs each agent key → *"agent A acts for tenant T, scope S"*
  - agent key → signs the fact (the per-action leg above)

  Verifying authority = walk `agent ← tenant ← platform-root` and check the fact's namespace matches
  the tenant's scope. The trust anchor is the platform root public key.

## Where the delegation certs live: the verifier's trust store (not the envelope)

The agent→action signature **must** be in the envelope (it's per-fact). The two delegation certs
(platform→tenant, tenant→agent) change **rarely**, so they live in the **verifier's local
`TrustStore`**, not in every envelope. This keeps envelopes small **and stays offline** — the
threat-model promise ("verifiable from the artifact alone, no network") is preserved because the trust
store is local state, not a network call. Verification needs: the envelope + the subject id + the
local trust store. No network, ever.

Envelopes **may** still inline the delegation certs as extra `Attestation`s (roles `tenant-namespace`
/ `platform-runtime`) when provenance must travel across a trust boundary self-contained — the
`Vec<Attestation>` already supports this. That's an option, not a requirement; the common path is
agent-signs-action in the envelope + delegation in the store.

## Roles (the `Attestation.role` string)

- `agent-signs-action` — the per-fact agent signature. **Always present** (attestation[0]).
- `tenant-namespace` — a tenant-root attestation; present when inlined for portability.
- `platform-runtime` — a platform-root attestation; present when inlined for portability.

## Verification algorithm (what the adapter implements)

Given an envelope, a `subject_id`, and a `TrustStore`:

1. The `VerifierRegistry`/`SchemeRegistry` has a verifier for `envelope.algorithm` — else
   `UnknownAlgorithm`. (Crypto-agility: Ed25519 today, ML-DSA/hybrid additive.)
2. At least one attestation — else `NoAttestations`.
3. The `agent-signs-action` signature verifies over `signed_payload(subject_id)` under the asserting
   agent's public key (resolved from the store) — else `BadSignature`.
4. **Authority:** the asserting agent chains `agent ← tenant ← platform-root` in the store, and the
   namespace embedded in `subject_id` / `asserting_agent` is within that tenant's scope — else the
   fact is well-signed but **unauthorized** (treat as untrusted).
5. **Revocation** is omission: a revoked agent/tenant key is dropped from the store; offline
   verification uses the store snapshot at verification time.

Steps 1–3 are what `Provenance::verify` does now. Step 4–5 are the `TrustStore`/delegation layer the
ember-crypto adapter adds.

## Crypto-agility (unchanged rule)

Every leg names its algorithm — the envelope header for the action signature, the cert records for the
delegation signatures — and dispatch goes through `ember-crypto`'s `SchemeRegistry`. An Ed25519 →
ML-DSA migration is additive and can proceed leg-by-leg (e.g. agent keys move to hybrid before tenant
roots do); old envelopes keep verifying.

## Consequence for the build

- **No envelope schema change** → the artifact format (item 4) serializes the final schema safely.
- The ember-crypto adapter implements: a `TrustStore` (identities → public keys + delegation edges +
  the platform root), an `EmberVerifier` (ember-graph's `Verifier` trait, backed by `ember-crypto` for
  step 3 and the store for step 4–5), and an `EmberSigner` (ember-graph's `Signer` trait wrapping an
  `ember-crypto` secret key).
- **SMB reconciliation:** the platform/tenant/agent keys map to the SMB platform's identity layer
  (`crates/identity`) and per-server tenant isolation. This doc is the ember-graph-side contract; the
  SMB side issues the certs (platform root in the platform, tenant roots per server, agent keys per
  running principal). Cross-referenced from the SMB knowledge-graph design.
