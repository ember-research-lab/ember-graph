//! Provenance envelope (v0.2 item 2) — **tamper-evident, offline-verifiable** assertion of who
//! claimed a fact, on what basis, over what source material.
//!
//! Threat model (the first consumer, SMB B3): *agents are employees; a compromised agent is a
//! compromised insider.* An edge asserted by a compromised agent is a poisoned fact every downstream
//! agent reasons from. So every extracted node/edge carries a [`Provenance`] envelope — the asserting
//! agent identity, the [`Confidence`](crate::Confidence) basis, a content-address of the source, and
//! a signature — **verifiable from the artifact alone, no network.**
//!
//! **Crypto-agility (the Ember rule):** the algorithm is named in the envelope header and dispatched
//! through an interface, never hardcoded. A [`VerifierRegistry`] holds multiple [`Verifier`]s at once,
//! so an Ed25519 → ML-DSA migration is *additive* — old envelopes keep verifying while new ones use
//! the new algorithm.
//!
//! **Zero-dep core, internal-reuse layer.** This module is the schema + the interface only — no
//! crypto here. The real [`Signer`]/[`Verifier`] implementations live in a feature-gated adapter that
//! **reuses cortex's Ed25519/BLAKE3 ledger primitives** (cortex-core `signing.rs` is entangled with
//! cortex-core's models/storage, so we *depend* rather than extract), with key custody through
//! ember-vault — ember-graph never touches raw key material.
//!
//! **Open dependency (flagged):** the SMB platform's three-way signing-key trust model (platform
//! attests runtime / tenant scopes namespace / agent signs action) is undecided. An ember-graph edge
//! signature is the *agent-signs-action* leg. The envelope below is built so the attestation chain is
//! **extensible** (`attestations: Vec<Attestation>`) rather than assuming a fixed shape — the final
//! schema is gated on that decision.

use std::collections::HashMap;

use crate::Confidence;

/// The basis on which an assertion was made — mirrors [`Confidence`], captured in the signed payload
/// so a verifier sees both *who* asserted and *how strongly*.
pub type Basis = Confidence;

/// One link in the attestation chain — an identity vouching for the assertion at some role. Kept a
/// `Vec` (not a fixed trio) so the SMB platform/tenant/agent trust model can slot in without a schema
/// break once it's decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attestation {
    /// The signing identity (e.g. an agent key id, a tenant scope, a platform runtime attestation).
    pub identity: String,
    /// What this identity is asserting by signing — consumer-defined role string
    /// ("agent-signs-action" | "tenant-namespace" | "platform-runtime" | …).
    pub role: String,
    /// Signature over [`Provenance::signed_payload`], by `identity`'s key.
    pub signature: Vec<u8>,
}

/// A tamper-evident provenance envelope carried by an extracted node or edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// Who asserted the fact (the primary agent identity).
    pub asserting_agent: String,
    /// The basis (EXTRACTED / INFERRED-tier / AMBIGUOUS).
    pub basis: Basis,
    /// Content address of the source material the assertion is drawn from (the consumer's hasher,
    /// e.g. BLAKE3 via the adapter). Lets a reviewer trace a node back to exact bytes.
    pub parent_hash: Vec<u8>,
    /// **Crypto-agility header** — the sign+hash algorithm id (e.g. "ed25519-blake3", "ml-dsa-65").
    /// The verifier dispatches on this; never assume a single algorithm.
    pub algorithm: String,
    /// The attestation chain (≥1; the asserting agent's own signature is the first). Extensible.
    pub attestations: Vec<Attestation>,
}

impl Provenance {
    /// The exact, deterministic bytes every attestation signs over. Includes the subject id (the
    /// edge/node this envelope is attached to) so a signature can't be lifted onto a different fact.
    pub fn signed_payload(&self, subject_id: &str) -> Vec<u8> {
        let mut p = Vec::new();
        // length-prefixed fields → unambiguous, no delimiter-injection.
        for field in [
            subject_id.as_bytes(),
            self.asserting_agent.as_bytes(),
            &[basis_tag(self.basis)],
            &self.parent_hash,
            self.algorithm.as_bytes(),
        ] {
            p.extend_from_slice(&(field.len() as u32).to_le_bytes());
            p.extend_from_slice(field);
        }
        p
    }

    /// Verify the envelope against `subject_id` using a registry of algorithm-keyed verifiers. Returns
    /// the [`VerifyOutcome`]. **Offline** — needs only the envelope, the subject id, and the trusted
    /// keys the verifiers hold. A flipped byte anywhere in the payload changes `signed_payload` and
    /// fails the signature.
    pub fn verify(&self, subject_id: &str, registry: &VerifierRegistry) -> VerifyOutcome {
        let Some(v) = registry.for_algorithm(&self.algorithm) else {
            return VerifyOutcome::UnknownAlgorithm;
        };
        if self.attestations.is_empty() {
            return VerifyOutcome::NoAttestations;
        }
        let payload = self.signed_payload(subject_id);
        for att in &self.attestations {
            if !v.verify(&payload, &att.signature, &att.identity) {
                return VerifyOutcome::BadSignature;
            }
        }
        VerifyOutcome::Valid
    }
}

/// Result of verifying a [`Provenance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyOutcome {
    Valid,
    /// A signature didn't verify (tamper, wrong key, or a revoked/untrusted identity).
    BadSignature,
    /// No verifier is registered for the envelope's algorithm.
    UnknownAlgorithm,
    /// The envelope carries no attestations.
    NoAttestations,
}

fn basis_tag(b: Basis) -> u8 {
    use crate::InferredTier::*;
    match b {
        Confidence::Extracted => 0,
        Confidence::Inferred(Strong) => 1,
        Confidence::Inferred(Clear) => 2,
        Confidence::Inferred(Reasonable) => 3,
        Confidence::Inferred(Weak) => 4,
        Confidence::Inferred(Tenuous) => 5,
        Confidence::Ambiguous => 6,
    }
}

/// Produces signatures for an algorithm (the real impl is the feature-gated cortex/vault adapter).
pub trait Signer {
    /// The algorithm id this signer writes into the envelope header.
    fn algorithm(&self) -> &str;
    /// Sign `payload` as `identity` (key custody is the signer's concern — never ember-graph's).
    fn sign(&self, payload: &[u8], identity: &str) -> Vec<u8>;
}

/// Verifies signatures for an algorithm. The verifier owns its trusted-keys view (so verification is
/// offline) and decides whether `identity` is trusted.
pub trait Verifier {
    fn algorithm(&self) -> &str;
    /// Does `signature` verify over `payload` as a trusted `identity`?
    fn verify(&self, payload: &[u8], signature: &[u8], identity: &str) -> bool;
}

/// A set of verifiers, one (or more) per algorithm — the crypto-agility seam. New algorithm = a new
/// verifier added here; old envelopes keep verifying. Boxed so heterogeneous algorithms coexist.
#[derive(Default)]
pub struct VerifierRegistry {
    verifiers: Vec<Box<dyn Verifier>>,
}

impl VerifierRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, v: Box<dyn Verifier>) -> Self {
        self.verifiers.push(v);
        self
    }
    pub fn for_algorithm(&self, algo: &str) -> Option<&dyn Verifier> {
        self.verifiers
            .iter()
            .find(|v| v.algorithm() == algo)
            .map(|b| b.as_ref())
    }
}

/// **O(1) lookup of provenance by subject id** — the graph-as-verification-substrate primitive.
///
/// An orchestrator works from *summaries* (cheap, but they drop nuance it can't see). To verify a
/// summary's claim, it pulls the **signed fact behind the claim** from the graph and checks against
/// it — not against the summary it came from. That retrieval must be cheap: the artifact's on-disk
/// provenance is a sorted `Vec` (canonical, content-addressable), but at runtime a verifier needs
/// **O(1)** access by subject. This index provides it. A subject may carry more than one envelope
/// (e.g. asserted by several agents), so each maps to a slice.
#[derive(Clone, Debug, Default)]
pub struct ProvenanceIndex {
    by_subject: HashMap<String, Vec<Provenance>>,
}

impl ProvenanceIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a `(subject_id, envelope)` pair.
    pub fn insert(&mut self, subject: impl Into<String>, envelope: Provenance) {
        self.by_subject
            .entry(subject.into())
            .or_default()
            .push(envelope);
    }

    /// Build the index from `(subject, envelope)` pairs (e.g. an [`crate::Artifact`]'s side-table).
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Provenance)>) -> Self {
        let mut ix = Self::new();
        for (subject, env) in pairs {
            ix.insert(subject, env);
        }
        ix
    }

    /// All envelopes attesting `subject`, in **O(1)** — an empty slice if none. This is the
    /// "fetch the signed fact behind this claim" call; works for node *and* edge subject ids.
    pub fn for_subject(&self, subject: &str) -> &[Provenance] {
        self.by_subject
            .get(subject)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total envelopes indexed (across all subjects).
    pub fn len(&self) -> usize {
        self.by_subject.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_subject.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferredTier;

    /// A test-only signer/verifier (NOT crypto — a keyed digest) to exercise the schema + the
    /// tamper-detection + crypto-agility flow without pulling a crypto dep into the zero-dep core.
    struct MockAlgo;
    fn mock_sig(payload: &[u8], identity: &str) -> Vec<u8> {
        let mut h: u64 = 1469598103934665603;
        for b in identity.bytes().chain(payload.iter().copied()) {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        h.to_le_bytes().to_vec()
    }
    impl Signer for MockAlgo {
        fn algorithm(&self) -> &str {
            "mock-v1"
        }
        fn sign(&self, payload: &[u8], identity: &str) -> Vec<u8> {
            mock_sig(payload, identity)
        }
    }
    impl Verifier for MockAlgo {
        fn algorithm(&self) -> &str {
            "mock-v1"
        }
        fn verify(&self, payload: &[u8], signature: &[u8], identity: &str) -> bool {
            mock_sig(payload, identity) == signature
        }
    }

    fn envelope(signer: &MockAlgo, subject: &str) -> Provenance {
        let mut p = Provenance {
            asserting_agent: "acme:receptionist".into(),
            basis: Confidence::Inferred(InferredTier::Clear),
            parent_hash: vec![0xde, 0xad, 0xbe, 0xef],
            algorithm: Signer::algorithm(signer).into(),
            attestations: vec![],
        };
        let sig = signer.sign(&p.signed_payload(subject), "acme:receptionist");
        p.attestations.push(Attestation {
            identity: "acme:receptionist".into(),
            role: "agent-signs-action".into(),
            signature: sig,
        });
        p
    }

    #[test]
    fn valid_envelope_verifies() {
        let signer = MockAlgo;
        let reg = VerifierRegistry::new().with(Box::new(MockAlgo));
        let p = envelope(&signer, "edge:john->acme");
        assert_eq!(p.verify("edge:john->acme", &reg), VerifyOutcome::Valid);
    }

    #[test]
    fn tampering_any_field_fails_verification() {
        let signer = MockAlgo;
        let reg = VerifierRegistry::new().with(Box::new(MockAlgo));

        // tamper the basis (a compromised agent upgrading INFERRED → EXTRACTED) → fails.
        let mut p = envelope(&signer, "edge:x");
        p.basis = Confidence::Extracted;
        assert_eq!(p.verify("edge:x", &reg), VerifyOutcome::BadSignature);

        // tamper the parent hash → fails.
        let mut p = envelope(&signer, "edge:x");
        p.parent_hash[0] ^= 0x01;
        assert_eq!(p.verify("edge:x", &reg), VerifyOutcome::BadSignature);

        // lift the signature onto a different subject → fails (subject is in the payload).
        let p = envelope(&signer, "edge:x");
        assert_eq!(
            p.verify("edge:DIFFERENT", &reg),
            VerifyOutcome::BadSignature
        );
    }

    #[test]
    fn unknown_algorithm_is_distinguished() {
        let signer = MockAlgo;
        let empty = VerifierRegistry::new(); // no verifier for "mock-v1"
        let p = envelope(&signer, "e");
        assert_eq!(p.verify("e", &empty), VerifyOutcome::UnknownAlgorithm);
    }

    #[test]
    fn crypto_agility_multiple_algorithms_coexist() {
        // A registry can hold several algorithms at once (Ed25519 → ML-DSA migration is additive).
        struct OtherAlgo;
        impl Verifier for OtherAlgo {
            fn algorithm(&self) -> &str {
                "future-pqc"
            }
            fn verify(&self, _: &[u8], _: &[u8], _: &str) -> bool {
                false
            }
        }
        let reg = VerifierRegistry::new()
            .with(Box::new(MockAlgo))
            .with(Box::new(OtherAlgo));
        assert!(reg.for_algorithm("mock-v1").is_some());
        assert!(reg.for_algorithm("future-pqc").is_some());
        assert!(reg.for_algorithm("nope").is_none());
    }

    #[test]
    fn provenance_index_retrieves_and_verifies_in_one_hop() {
        // The verification-substrate flow: index envelopes, retrieve the signed fact behind a
        // subject by key, then verify it — no scan of the corpus.
        let signer = MockAlgo;
        let reg = VerifierRegistry::new().with(Box::new(MockAlgo));
        let mut ix = ProvenanceIndex::new();
        ix.insert("edge:john->acme", envelope(&signer, "edge:john->acme"));
        ix.insert(
            "edge:acme->invoice42",
            envelope(&signer, "edge:acme->invoice42"),
        );

        // Unknown subject → empty (not a panic, not a scan).
        assert!(ix.for_subject("edge:nope").is_empty());

        // Known subject → the signed envelope, retrieved by key, which verifies against ground truth.
        let got = ix.for_subject("edge:john->acme");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].verify("edge:john->acme", &reg), VerifyOutcome::Valid);
        assert_eq!(ix.len(), 2);
    }
}
