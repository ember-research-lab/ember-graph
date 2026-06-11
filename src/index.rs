//! Serialization index — the **stable u64 tensor-address layer** (v0.2 item 1).
//!
//! [`NodeId`](crate::NodeId) is *identity* (content-derived, the join key). It can't give the GNN
//! consumer dense, ordered, stable integer indices — a content hash has no ordering and no tensor
//! address. This layer adds a persisted `canonical → u64` map, **assigned monotonically**, **never
//! reused, never recompacted**:
//!
//! - The u64 is **not identity** — it's a stable address into a tensor / feature matrix.
//! - A node interned once keeps its u64 across every serialize cycle (so tensor rows stay aligned).
//! - A deleted node **tombstones** its u64 — the slot is dead, never reassigned (a GNN masks it).
//! - A canonical **merge** records u64 lineage (`from_u64 → into_u64`) so tensor data addressed to
//!   the merged-away node redirects to the survivor instead of dangling.
//!
//! Pure, zero-dep, deterministic. Persisted as part of the artifact (item 4).

use std::collections::HashMap;

use crate::NodeId;

/// The persisted `canonical ↔ u64` index. Monotonic, append-only in spirit (u64s are never reused).
#[derive(Clone, Debug, Default)]
pub struct NodeIndex {
    /// Next u64 to hand out — only ever increases.
    next: u64,
    /// canonical id → its stable u64 (live nodes only).
    by_id: HashMap<NodeId, u64>,
    /// u64 → canonical id (reverse, live nodes only).
    by_u64: HashMap<u64, NodeId>,
    /// u64s that are dead (deleted or merged away) — never reassigned; a consumer masks these.
    tombstoned: Vec<u64>,
    /// u64 lineage for canonical merges: a merged-away u64 → the survivor's u64. Chains are followed
    /// by [`resolve`](NodeIndex::resolve).
    merged_into: HashMap<u64, u64>,
}

impl NodeIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an id: return its existing u64, or assign the next monotonic one. The only place a
    /// u64 is minted. Stable — interning the same id again returns the same u64.
    pub fn intern(&mut self, id: &NodeId) -> u64 {
        if let Some(&n) = self.by_id.get(id) {
            return n;
        }
        let n = self.next;
        self.next += 1;
        self.by_id.insert(id.clone(), n);
        self.by_u64.insert(n, id.clone());
        n
    }

    /// The u64 for a live id, if interned.
    pub fn u64_of(&self, id: &NodeId) -> Option<u64> {
        self.by_id.get(id).copied()
    }

    /// The id for a live u64, if any.
    pub fn id_of(&self, n: u64) -> Option<&NodeId> {
        self.by_u64.get(&n)
    }

    /// Is this u64 dead (tombstoned or merged away)?
    pub fn is_tombstoned(&self, n: u64) -> bool {
        self.tombstoned.contains(&n)
    }

    /// All dead u64s (for masking in a tensor consumer).
    pub fn tombstones(&self) -> &[u64] {
        &self.tombstoned
    }

    /// Delete a node: free its canonical mapping and **tombstone its u64** (never reused). No-op if
    /// the id was never interned. Returns the freed u64.
    pub fn tombstone(&mut self, id: &NodeId) -> Option<u64> {
        let n = self.by_id.remove(id)?;
        self.by_u64.remove(&n);
        self.tombstoned.push(n);
        Some(n)
    }

    /// Record a canonical merge of `from` into `into`: `into` becomes the survivor, `from`'s u64 is
    /// tombstoned and its lineage recorded (`from_u64 → into_u64`) so addresses to `from` redirect.
    /// Both are interned first if needed. Returns `(from_u64, into_u64)`.
    pub fn merge(&mut self, from: &NodeId, into: &NodeId) -> (u64, u64) {
        let into_u64 = self.intern(into);
        let from_u64 = self.intern(from);
        if from_u64 != into_u64 {
            self.by_id.remove(from);
            self.by_u64.remove(&from_u64);
            self.tombstoned.push(from_u64);
            self.merged_into.insert(from_u64, into_u64);
        }
        (from_u64, into_u64)
    }

    /// Follow merge lineage to the surviving u64 (identity if `n` was never merged away). A tensor
    /// row addressed to a merged-away node resolves here to the survivor's row.
    pub fn resolve(&self, mut n: u64) -> u64 {
        let mut guard = 0;
        while let Some(&into) = self.merged_into.get(&n) {
            n = into;
            guard += 1;
            if guard > self.merged_into.len() {
                break; // defensive: cycles can't form via intern order, but never loop forever
            }
        }
        n
    }

    /// Count of live (non-tombstoned) interned nodes.
    pub fn live_count(&self) -> usize {
        self.by_id.len()
    }

    /// The highest u64 ever assigned + 1 — the dense width a tensor must allocate (including dead
    /// slots, which a consumer masks).
    pub fn capacity(&self) -> u64 {
        self.next
    }

    /// Snapshot the raw internal state for serialization (item 4). The fields are private and the
    /// `artifact` module is a sibling that can't reach them, so this `pub(crate)` accessor is the
    /// only sanctioned door — it clones the five fields verbatim so a round-trip is byte-faithful.
    pub(crate) fn snapshot(&self) -> NodeIndexSnapshot {
        NodeIndexSnapshot {
            next: self.next,
            by_id: self.by_id.clone(),
            by_u64: self.by_u64.clone(),
            tombstoned: self.tombstoned.clone(),
            merged_into: self.merged_into.clone(),
        }
    }

    /// Reconstruct a `NodeIndex` from a [`NodeIndexSnapshot`] (item 4 deserialization). Inverse of
    /// [`snapshot`](NodeIndex::snapshot): restores the five fields exactly, so `capacity`, the live
    /// maps, the tombstones, and the merge lineage all come back identical.
    pub(crate) fn from_snapshot(s: NodeIndexSnapshot) -> Self {
        NodeIndex {
            next: s.next,
            by_id: s.by_id,
            by_u64: s.by_u64,
            tombstoned: s.tombstoned,
            merged_into: s.merged_into,
        }
    }
}

/// A verbatim copy of [`NodeIndex`]'s five private fields, for the serializer in the `artifact`
/// module (a sibling that can't otherwise reach them). Crate-internal by design.
pub(crate) struct NodeIndexSnapshot {
    pub(crate) next: u64,
    pub(crate) by_id: HashMap<NodeId, u64>,
    pub(crate) by_u64: HashMap<u64, NodeId>,
    pub(crate) tombstoned: Vec<u64>,
    pub(crate) merged_into: HashMap<u64, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> NodeId {
        NodeId(s.into())
    }

    #[test]
    fn intern_is_stable_and_monotonic() {
        let mut ix = NodeIndex::new();
        let a = ix.intern(&id("a"));
        let b = ix.intern(&id("b"));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        // Re-interning returns the same u64 (stable across "serialize cycles").
        assert_eq!(ix.intern(&id("a")), a);
        assert_eq!(ix.intern(&id("b")), b);
        // A new node gets the next monotonic id.
        assert_eq!(ix.intern(&id("c")), 2);
        assert_eq!(ix.capacity(), 3);
    }

    #[test]
    fn tombstone_never_reuses_the_u64() {
        let mut ix = NodeIndex::new();
        let a = ix.intern(&id("a"));
        ix.intern(&id("b"));
        let freed = ix.tombstone(&id("a")).unwrap();
        assert_eq!(freed, a);
        assert!(ix.is_tombstoned(a));
        assert_eq!(ix.u64_of(&id("a")), None); // canonical mapping gone
                                               // A brand-new node does NOT reuse a's slot — next is monotonic.
        let c = ix.intern(&id("c"));
        assert_ne!(c, a);
        assert_eq!(c, 2);
    }

    #[test]
    fn merge_keeps_u64_lineage() {
        let mut ix = NodeIndex::new();
        let from = ix.intern(&id("dup"));
        let into = ix.intern(&id("canon"));
        let (f, t) = ix.merge(&id("dup"), &id("canon"));
        assert_eq!((f, t), (from, into));
        // The merged-away u64 resolves to the survivor — tensor rows redirect, never dangle.
        assert_eq!(ix.resolve(from), into);
        assert!(ix.is_tombstoned(from));
        assert_eq!(ix.u64_of(&id("dup")), None);
        assert_eq!(ix.u64_of(&id("canon")), Some(into));
        // Survivor's own u64 resolves to itself.
        assert_eq!(ix.resolve(into), into);
    }

    #[test]
    fn survivors_keep_their_u64_across_churn() {
        // The "u64 stability across serialize cycles" property: add/delete/merge, survivors unchanged.
        let mut ix = NodeIndex::new();
        let keep = ix.intern(&id("keep"));
        ix.intern(&id("del"));
        let other = ix.intern(&id("other"));
        ix.tombstone(&id("del"));
        ix.intern(&id("new1"));
        ix.merge(&id("other"), &id("keep")); // other merges into keep
                                             // keep is untouched; other resolves to keep; new ids climbed past the dead slots.
        assert_eq!(ix.u64_of(&id("keep")), Some(keep));
        assert_eq!(ix.resolve(other), keep);
        assert_eq!(ix.intern(&id("new2")), ix.capacity() - 1);
    }
}
