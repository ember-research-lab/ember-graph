//! crypto_adapter — real [`Signer`]/[`Verifier`] for the provenance envelope, backed by `ember-crypto`.
//!
//! The envelope schema ([`Provenance`]) and the verification *interface* live in [`provenance`]; this
//! module is the feature-gated bridge that makes them **cryptographically real** using the org's
//! shared crypto surface (`ember-crypto`: Ed25519 today, ML-DSA-65 / hybrid additive). It implements
//! the trust model in `design/trust-model.md`:
//!
//! - **Integrity** — the agent's signature over [`Provenance::signed_payload`], checked via
//!   `ember-crypto`. This is the `agent-signs-action` leg already in the envelope.
//! - **Authority** — a delegation chain `agent ← tenant ← platform-root` held in a [`TrustStore`],
//!   plus a namespace-scope check. The platform root public key is the offline trust anchor.
//!
//! Verification stays **offline**: it needs the envelope, the subject id, and the local trust store —
//! no network. Adding ML-DSA / hybrid is additive (the [`SchemeRegistry`] dispatches on the named
//! algorithm); old envelopes keep verifying.

use std::collections::HashMap;
use std::sync::Arc;

use ember_crypto::{
    alg, Ed25519Scheme, HybridScheme, MlDsa65Scheme, PublicKey as EcPublicKey,
    SecretKey as EcSecretKey, Signature as EcSignature, SignatureScheme,
};

use crate::provenance::{Signer, Verifier, VerifierRegistry};
use crate::{Provenance, VerifyOutcome};

/// The ember-crypto scheme for an algorithm id, or `None` if unknown. Single dispatch point so the
/// signer and verifier can't disagree on which algorithm maps to which scheme.
fn scheme_for(algorithm: &str) -> Option<Box<dyn SignatureScheme>> {
    match algorithm {
        alg::ED25519 => Some(Box::new(Ed25519Scheme)),
        alg::ML_DSA_65 => Some(Box::new(MlDsa65Scheme)),
        alg::HYBRID_ED25519_ML_DSA_65 => Some(Box::new(HybridScheme)),
        _ => None,
    }
}

/// The verifier's view of who is trusted and how identities chain to the platform root. Small,
/// slowly-changing, **local** state — so verification needs no network. Revocation is omission: drop a
/// key and it (and everything below it) stops verifying.
#[derive(Clone, Default)]
pub struct TrustStore {
    /// The trust anchor — the platform root identity.
    platform_root: String,
    /// identity → its public key (the key carries its own algorithm).
    keys: HashMap<String, EcPublicKey>,
    /// delegation edges, child → parent (agent → tenant, tenant → platform-root).
    parent: HashMap<String, String>,
    /// tenant identity → the namespace it owns (e.g. `"tenant7"`).
    scope: HashMap<String, String>,
}

impl TrustStore {
    /// New store anchored at a platform root identity + its public key.
    pub fn new(platform_root: impl Into<String>, root_key: EcPublicKey) -> Self {
        let platform_root = platform_root.into();
        let mut keys = HashMap::new();
        keys.insert(platform_root.clone(), root_key);
        Self {
            platform_root,
            keys,
            parent: HashMap::new(),
            scope: HashMap::new(),
        }
    }

    /// Register a tenant: its key, its delegation to the platform root, and the namespace it owns.
    pub fn add_tenant(
        &mut self,
        tenant: impl Into<String>,
        key: EcPublicKey,
        namespace: impl Into<String>,
    ) {
        let tenant = tenant.into();
        self.parent
            .insert(tenant.clone(), self.platform_root.clone());
        self.scope.insert(tenant.clone(), namespace.into());
        self.keys.insert(tenant, key);
    }

    /// Register an agent: its key and its delegation to a tenant.
    pub fn add_agent(
        &mut self,
        agent: impl Into<String>,
        key: EcPublicKey,
        tenant: impl Into<String>,
    ) {
        let agent = agent.into();
        self.parent.insert(agent.clone(), tenant.into());
        self.keys.insert(agent, key);
    }

    /// The public key registered for an identity, if any.
    pub fn public_key(&self, identity: &str) -> Option<&EcPublicKey> {
        self.keys.get(identity)
    }

    /// Does `identity` chain `… ← platform-root` through the delegation edges? (Cycle-guarded.)
    pub fn chains_to_root(&self, identity: &str) -> bool {
        if !self.keys.contains_key(identity) {
            return false;
        }
        let mut cur = identity;
        let mut hops = 0;
        while cur != self.platform_root {
            let Some(p) = self.parent.get(cur) else {
                return false; // dangling — not delegated up to the root
            };
            if !self.keys.contains_key(p) {
                return false; // parent key not trusted (e.g. revoked)
            }
            cur = p;
            hops += 1;
            if hops > self.parent.len() + 1 {
                return false; // defensive cycle break
            }
        }
        true
    }

    /// Is `identity` authorized to assert a fact in `namespace`? It must chain to the root **and** own
    /// (directly, if it's a tenant) or belong to a tenant that owns `namespace`.
    pub fn is_authorized(&self, identity: &str, namespace: &str) -> bool {
        if !self.chains_to_root(identity) {
            return false;
        }
        // The owning scope: the identity's own (if it's a tenant) or its parent tenant's.
        let owner_scope = self
            .scope
            .get(identity)
            .or_else(|| self.parent.get(identity).and_then(|p| self.scope.get(p)));
        owner_scope.map(|ns| ns == namespace).unwrap_or(false)
    }
}

/// Signs as one identity, wrapping that identity's `ember-crypto` secret key. (The `Signer::sign`
/// `identity` argument is informational — the wrapped key *is* the identity's; a mismatch is the
/// caller's bug.) Fails closed: an unusable key yields an empty signature, which fails verification.
pub struct EmberSigner {
    secret: EcSecretKey,
}

impl EmberSigner {
    pub fn new(secret: EcSecretKey) -> Self {
        Self { secret }
    }
}

impl Signer for EmberSigner {
    fn algorithm(&self) -> &str {
        self.secret.algorithm
    }

    fn sign(&self, payload: &[u8], _identity: &str) -> Vec<u8> {
        scheme_for(self.secret.algorithm)
            .and_then(|s| s.sign(&self.secret, payload).ok())
            .map(|sig| sig.bytes)
            .unwrap_or_default()
    }
}

/// Verifies one algorithm's signatures against a shared [`TrustStore`]. One of these is registered per
/// algorithm in a [`VerifierRegistry`] (the crypto-agility seam). `verify` enforces **integrity**
/// (the signature checks out under the identity's key) **and membership** (the identity chains to the
/// platform root). The namespace-scope leg is [`verify_authorized`] (it needs the subject's namespace,
/// which the bare `Verifier::verify` interface doesn't carry).
pub struct EmberVerifier {
    algorithm: &'static str,
    trust: Arc<TrustStore>,
}

impl EmberVerifier {
    pub fn new(algorithm: &'static str, trust: Arc<TrustStore>) -> Self {
        Self { algorithm, trust }
    }
}

impl Verifier for EmberVerifier {
    fn algorithm(&self) -> &str {
        self.algorithm
    }

    fn verify(&self, payload: &[u8], signature: &[u8], identity: &str) -> bool {
        let Some(public) = self.trust.public_key(identity) else {
            return false; // unknown identity
        };
        if public.algorithm != self.algorithm {
            return false; // key is for a different algorithm
        }
        let Some(scheme) = scheme_for(self.algorithm) else {
            return false;
        };
        let sig = EcSignature {
            algorithm: self.algorithm,
            bytes: signature.to_vec(),
        };
        if scheme.verify(public, payload, &sig).is_err() {
            return false; // integrity failed
        }
        // Integrity ok — the identity must also chain to the platform root.
        self.trust.chains_to_root(identity)
    }
}

/// Build a [`VerifierRegistry`] wired to a shared trust store, with a verifier per supported algorithm
/// (Ed25519 + ML-DSA-65 + hybrid). This is what a consumer passes to [`Provenance::verify`].
pub fn verifier_registry(trust: Arc<TrustStore>) -> VerifierRegistry {
    VerifierRegistry::new()
        .with(Box::new(EmberVerifier::new(alg::ED25519, trust.clone())))
        .with(Box::new(EmberVerifier::new(alg::ML_DSA_65, trust.clone())))
        .with(Box::new(EmberVerifier::new(
            alg::HYBRID_ED25519_ML_DSA_65,
            trust,
        )))
}

/// The full trust-model verdict for an envelope: integrity + membership (via the registry) **and**
/// namespace authority (via the trust store). The complete steps 1–5 of `design/trust-model.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorizedOutcome {
    /// Integrity + membership + namespace authority all hold.
    Authorized,
    /// Cryptographically valid, but the agent isn't authorized for this namespace.
    Unauthorized,
    /// A signature didn't verify, the identity is unknown, or it doesn't chain to the root.
    BadSignature,
    /// No verifier registered for the envelope's algorithm.
    UnknownAlgorithm,
    /// The envelope carries no attestations.
    NoAttestations,
}

/// Verify an envelope **and** that its asserting agent is authorized for `namespace`. `subject_id` is
/// the node/edge the envelope is attached to (bound into the signed payload); `namespace` is the
/// data namespace the fact belongs to (e.g. the `tenant7` in a `tenant7::…` [`crate::NodeId`]).
pub fn verify_authorized(
    prov: &Provenance,
    subject_id: &str,
    namespace: &str,
    registry: &VerifierRegistry,
    trust: &TrustStore,
) -> AuthorizedOutcome {
    match prov.verify(subject_id, registry) {
        VerifyOutcome::Valid => {}
        VerifyOutcome::BadSignature => return AuthorizedOutcome::BadSignature,
        VerifyOutcome::UnknownAlgorithm => return AuthorizedOutcome::UnknownAlgorithm,
        VerifyOutcome::NoAttestations => return AuthorizedOutcome::NoAttestations,
    }
    if trust.is_authorized(&prov.asserting_agent, namespace) {
        AuthorizedOutcome::Authorized
    } else {
        AuthorizedOutcome::Unauthorized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::Attestation;
    use crate::Confidence;

    /// Build (trust store, agent identity, agent signer) for one algorithm, wired
    /// platform-root → tenant("tenant7") → agent. Returns the pieces a test needs.
    fn wired(scheme: &dyn SignatureScheme) -> (TrustStore, String, EmberSigner) {
        let root = scheme.generate().unwrap();
        let tenant = scheme.generate().unwrap();
        let agent = scheme.generate().unwrap();
        let (root_id, tenant_id, agent_id) = (
            "platform:root".to_string(),
            "tenant7:root".to_string(),
            "tenant7:agent:reception".to_string(),
        );
        let mut store = TrustStore::new(root_id, root.public);
        store.add_tenant(tenant_id.clone(), tenant.public, "tenant7");
        store.add_agent(agent_id.clone(), agent.public, tenant_id);
        (store, agent_id, EmberSigner::new(agent.secret))
    }

    /// Sign a fresh envelope as `agent_id` over `subject`.
    fn signed_envelope(signer: &EmberSigner, agent_id: &str, subject: &str) -> Provenance {
        let mut prov = Provenance {
            asserting_agent: agent_id.to_string(),
            basis: Confidence::Extracted,
            parent_hash: vec![0xab, 0xcd, 0xef],
            algorithm: signer.algorithm().to_string(),
            attestations: vec![],
        };
        let sig = signer.sign(&prov.signed_payload(subject), agent_id);
        prov.attestations.push(Attestation {
            identity: agent_id.to_string(),
            role: "agent-signs-action".to_string(),
            signature: sig,
        });
        prov
    }

    #[test]
    fn ed25519_valid_and_authorized() {
        let (store, agent, signer) = wired(&Ed25519Scheme);
        let subject = "tenant7::acme_llc";
        let prov = signed_envelope(&signer, &agent, subject);
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());

        // integrity + membership
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::Valid);
        // full trust-model verdict
        assert_eq!(
            verify_authorized(&prov, subject, "tenant7", &reg, &trust),
            AuthorizedOutcome::Authorized
        );
    }

    #[test]
    fn wrong_namespace_is_unauthorized_though_signature_is_valid() {
        let (store, agent, signer) = wired(&Ed25519Scheme);
        let subject = "tenant7::acme_llc";
        let prov = signed_envelope(&signer, &agent, subject);
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        // Signature is valid, but the agent's tenant owns "tenant7", not "tenant9".
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::Valid);
        assert_eq!(
            verify_authorized(&prov, subject, "tenant9", &reg, &trust),
            AuthorizedOutcome::Unauthorized
        );
    }

    #[test]
    fn tampered_basis_fails_signature() {
        let (store, agent, signer) = wired(&Ed25519Scheme);
        let subject = "tenant7::acme_llc";
        let mut prov = signed_envelope(&signer, &agent, subject);
        prov.basis = Confidence::Ambiguous; // a compromised agent downgrading/upgrading the basis
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::BadSignature);
        assert_eq!(
            verify_authorized(&prov, subject, "tenant7", &reg, &trust),
            AuthorizedOutcome::BadSignature
        );
    }

    #[test]
    fn unknown_agent_is_rejected() {
        let (store, _agent, signer) = wired(&Ed25519Scheme);
        let subject = "tenant7::acme_llc";
        // Sign as an identity that is NOT in the trust store.
        let prov = signed_envelope(&signer, "tenant7:agent:impostor", subject);
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::BadSignature);
    }

    #[test]
    fn lifted_signature_to_other_subject_fails() {
        let (store, agent, signer) = wired(&Ed25519Scheme);
        let prov = signed_envelope(&signer, &agent, "tenant7::acme_llc");
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        // The subject is bound into the signed payload — verifying against a different subject fails.
        assert_eq!(
            prov.verify("tenant7::other_corp", &reg),
            VerifyOutcome::BadSignature
        );
    }

    #[test]
    fn hybrid_pqc_path_valid_and_authorized() {
        // Post-quantum migration posture: the agent signs hybrid (Ed25519 + ML-DSA-65).
        let (store, agent, signer) = wired(&HybridScheme);
        let subject = "tenant7::acme_llc";
        let prov = signed_envelope(&signer, &agent, subject);
        assert_eq!(prov.algorithm, alg::HYBRID_ED25519_ML_DSA_65);
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::Valid);
        assert_eq!(
            verify_authorized(&prov, subject, "tenant7", &reg, &trust),
            AuthorizedOutcome::Authorized
        );
    }

    #[test]
    fn ml_dsa_path_verifies() {
        let (store, agent, signer) = wired(&MlDsa65Scheme);
        let subject = "tenant7::acme_llc";
        let prov = signed_envelope(&signer, &agent, subject);
        assert_eq!(prov.algorithm, alg::ML_DSA_65);
        let trust = Arc::new(store);
        let reg = verifier_registry(trust.clone());
        assert_eq!(prov.verify(subject, &reg), VerifyOutcome::Valid);
    }
}
