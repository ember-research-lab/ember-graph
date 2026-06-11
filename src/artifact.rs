//! artifact — a single, versioned, self-contained serialization of a graph (v0.2 item 4).
//!
//! An [`Artifact`] bundles, by value, everything needed to reconstitute a knowledge graph offline:
//! the [`Graph`](crate::Graph) (nodes + edges), the [`NodeIndex`](crate::NodeIndex) (the stable u64
//! tensor-address layer), and a provenance side-table (`subject_id → `[`Provenance`](crate::Provenance)
//! envelopes — the graph doesn't store provenance itself, so signed facts travel alongside it here).
//!
//! **Container shape.** [`to_bytes`](Artifact::to_bytes) emits a **flat, single-read,
//! section-framed** byte container — an 8-byte magic, a `u32` format version, then four
//! length-prefixed sections (nodes, edges, index, provenance) read in one forward pass. Every
//! variable-length field and section is `[u32 LE len][bytes]`-framed (the same discipline as
//! [`Provenance::signed_payload`](crate::Provenance::signed_payload)), so parsing is unambiguous and
//! injection-free. This is *not* a zero-copy mmap format — there is no alignment, no fixed-offset
//! table, no in-place field access; you read it start-to-finish into owned structures.
//!
//! **Deterministic output.** `to_bytes` is reproducible: nodes are emitted sorted by id, edges by
//! `(source, target, relation)`, provenance entries by subject, and the index maps in sorted key
//! order. Equal logical content yields byte-identical output regardless of insertion order — the
//! property a content-addressing scheme needs.
//!
//! **Strict parsing.** [`from_bytes`](Artifact::from_bytes) rejects bad magic, an unknown version,
//! truncation, trailing bytes, a bad enum tag, and bad UTF-8 — round-tripping `to_bytes` exactly.
//!
//! [`to_json`](Artifact::to_json) is a one-way, hand-rolled, human-readable debug export (it is *not*
//! required to parse back). Zero third-party dependencies throughout — every encoder is hand-rolled.

use std::collections::HashMap;
use std::fmt;

use crate::index::NodeIndexSnapshot;
use crate::provenance::Attestation;
use crate::{
    Confidence, Edge, Graph, InferredTier, Node, NodeId, NodeIndex, Provenance, ProvenanceIndex,
};

/// The 8-byte container magic.
const MAGIC: &[u8; 8] = b"EMGRAPH\0";
/// The format version `to_bytes` writes and `from_bytes` accepts.
const FORMAT_VERSION: u32 = 1;

/// A self-contained, versioned bundle of a graph + its index + a provenance side-table.
///
/// Construct with [`Artifact::new`]; attach signed facts with [`Artifact::with_provenance`] /
/// [`Artifact::attach_provenance`]. Serialize with [`to_bytes`](Artifact::to_bytes) and restore with
/// [`from_bytes`](Artifact::from_bytes).
pub struct Artifact {
    /// The graph (nodes + directed adjacency).
    pub graph: Graph,
    /// The stable u64 tensor-address index.
    pub index: NodeIndex,
    /// Signed-fact side-table: `(subject_id, envelope)`. Often empty. A subject may carry more than
    /// one envelope, so this is a `Vec` of pairs rather than a map.
    pub provenance: Vec<(String, Provenance)>,
}

impl Artifact {
    /// Bundle a graph and an index with an empty provenance side-table.
    pub fn new(graph: Graph, index: NodeIndex) -> Self {
        Artifact {
            graph,
            index,
            provenance: Vec::new(),
        }
    }

    /// Attach a provenance envelope for `subject_id`, consuming and returning `self` (builder style).
    pub fn with_provenance(mut self, subject_id: impl Into<String>, envelope: Provenance) -> Self {
        self.provenance.push((subject_id.into(), envelope));
        self
    }

    /// Attach a provenance envelope for `subject_id` in place.
    pub fn attach_provenance(&mut self, subject_id: impl Into<String>, envelope: Provenance) {
        self.provenance.push((subject_id.into(), envelope));
    }

    /// Build the **O(1)** [`ProvenanceIndex`] from this artifact's side-table — the runtime structure
    /// a verifier queries (the on-disk form is the canonical sorted `Vec`; this is its indexed view).
    pub fn provenance_index(&self) -> ProvenanceIndex {
        ProvenanceIndex::from_pairs(self.provenance.iter().cloned())
    }

    /// Serialize to the flat, section-framed, deterministic byte container (see the module docs).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());

        // Four length-prefixed sections, in order. Each is built independently then framed, so the
        // container reads as one forward pass with no back-references.
        write_section(&mut out, &self.encode_nodes());
        write_section(&mut out, &self.encode_edges());
        write_section(&mut out, &self.encode_index());
        write_section(&mut out, &self.encode_provenance());
        out
    }

    /// Parse a container produced by [`to_bytes`](Artifact::to_bytes). Strict: see [`ArtifactError`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Artifact, ArtifactError> {
        let mut r = Reader::new(bytes);
        let magic = r.take(MAGIC.len())?;
        if magic != MAGIC {
            return Err(ArtifactError::BadMagic);
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(ArtifactError::UnsupportedVersion(version));
        }

        let nodes_sec = r.section()?;
        let edges_sec = r.section()?;
        let index_sec = r.section()?;
        let prov_sec = r.section()?;
        // Strict: no bytes may follow the last section.
        if !r.at_end() {
            return Err(ArtifactError::TrailingBytes);
        }

        let nodes = decode_nodes(nodes_sec)?;
        let edges = decode_edges(edges_sec)?;
        let index = decode_index(index_sec)?;
        let provenance = decode_provenance(prov_sec)?;

        // Reconstruct the Graph by inserting directly into its private fields (artifact is a
        // descendant module of the crate root, so it may). We bypass `add_edge`/`add_node` on
        // purpose: those mutate on conflict (keep-higher-confidence, fill-blank), which would not
        // reproduce the stored confidence/order faithfully. The serialized form is already the
        // canonical, deduplicated state.
        let mut graph = Graph::new();
        for node in nodes {
            graph.nodes.insert(node.id.clone(), node);
        }
        for edge in edges {
            graph.out.entry(edge.source.clone()).or_default().push(edge);
        }

        Ok(Artifact {
            graph,
            index,
            provenance,
        })
    }

    /// Hand-rolled, zero-dep, human-readable JSON debug export. One-way (not required to parse back).
    /// Byte fields (`parent_hash`, signatures) render as lowercase hex strings.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        s.push_str("{\n");
        s.push_str(&format!("  \"format_version\": {FORMAT_VERSION},\n"));

        // nodes (sorted by id, matching to_bytes order)
        let mut nodes: Vec<&Node> = self.graph.nodes.values().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        s.push_str("  \"nodes\": [\n");
        for (i, n) in nodes.iter().enumerate() {
            s.push_str("    {");
            s.push_str(&format!("\"id\": {}, ", json_str(&n.id.0)));
            s.push_str(&format!("\"label\": {}, ", json_str(&n.label)));
            s.push_str(&format!("\"kind\": {}, ", json_str(&n.kind)));
            match &n.source {
                Some(src) => s.push_str(&format!("\"source\": {}", json_str(src))),
                None => s.push_str("\"source\": null"),
            }
            s.push('}');
            s.push_str(if i + 1 < nodes.len() { ",\n" } else { "\n" });
        }
        s.push_str("  ],\n");

        // edges (sorted by (source, target, relation))
        let mut edges: Vec<&Edge> = self.graph.out.values().flatten().collect();
        edges.sort_by(|a, b| {
            (a.source.cmp(&b.source))
                .then(a.target.cmp(&b.target))
                .then(a.relation.cmp(&b.relation))
        });
        s.push_str("  \"edges\": [\n");
        for (i, e) in edges.iter().enumerate() {
            s.push_str("    {");
            s.push_str(&format!("\"source\": {}, ", json_str(&e.source.0)));
            s.push_str(&format!("\"target\": {}, ", json_str(&e.target.0)));
            s.push_str(&format!("\"relation\": {}, ", json_str(&e.relation)));
            s.push_str(&format!(
                "\"confidence\": {}",
                json_str(confidence_name(e.confidence))
            ));
            s.push('}');
            s.push_str(if i + 1 < edges.len() { ",\n" } else { "\n" });
        }
        s.push_str("  ],\n");

        // index
        let snap = self.index.snapshot();
        s.push_str("  \"index\": {\n");
        s.push_str(&format!("    \"capacity\": {},\n", snap.next));
        let mut by_id: Vec<(&NodeId, &u64)> = snap.by_id.iter().collect();
        by_id.sort_by(|a, b| a.0.cmp(b.0));
        s.push_str("    \"by_id\": {");
        for (i, (id, u)) in by_id.iter().enumerate() {
            s.push_str(&format!("{}: {}", json_str(&id.0), u));
            if i + 1 < by_id.len() {
                s.push_str(", ");
            }
        }
        s.push_str("},\n");
        let mut tombs = snap.tombstoned.clone();
        tombs.sort_unstable();
        let tomb_list: Vec<String> = tombs.iter().map(|t| t.to_string()).collect();
        s.push_str(&format!(
            "    \"tombstoned\": [{}],\n",
            tomb_list.join(", ")
        ));
        let mut merged: Vec<(&u64, &u64)> = snap.merged_into.iter().collect();
        merged.sort_by_key(|(k, _)| **k);
        s.push_str("    \"merged_into\": {");
        for (i, (from, into)) in merged.iter().enumerate() {
            s.push_str(&format!("\"{from}\": {into}"));
            if i + 1 < merged.len() {
                s.push_str(", ");
            }
        }
        s.push_str("}\n");
        s.push_str("  },\n");

        // provenance (sorted by subject)
        let mut prov: Vec<&(String, Provenance)> = self.provenance.iter().collect();
        prov.sort_by(|a, b| a.0.cmp(&b.0));
        s.push_str("  \"provenance\": [\n");
        for (i, (subject, p)) in prov.iter().enumerate() {
            s.push_str("    {");
            s.push_str(&format!("\"subject\": {}, ", json_str(subject)));
            s.push_str(&format!(
                "\"asserting_agent\": {}, ",
                json_str(&p.asserting_agent)
            ));
            s.push_str(&format!(
                "\"basis\": {}, ",
                json_str(confidence_name(p.basis))
            ));
            s.push_str(&format!(
                "\"parent_hash\": {}, ",
                json_str(&hex(&p.parent_hash))
            ));
            s.push_str(&format!("\"algorithm\": {}, ", json_str(&p.algorithm)));
            s.push_str("\"attestations\": [");
            for (j, att) in p.attestations.iter().enumerate() {
                s.push('{');
                s.push_str(&format!("\"identity\": {}, ", json_str(&att.identity)));
                s.push_str(&format!("\"role\": {}, ", json_str(&att.role)));
                s.push_str(&format!(
                    "\"signature\": {}",
                    json_str(&hex(&att.signature))
                ));
                s.push('}');
                if j + 1 < p.attestations.len() {
                    s.push_str(", ");
                }
            }
            s.push(']');
            s.push('}');
            s.push_str(if i + 1 < prov.len() { ",\n" } else { "\n" });
        }
        s.push_str("  ]\n");

        s.push_str("}\n");
        s
    }

    // --- section encoders (each returns the framed body of one section) ---

    fn encode_nodes(&self) -> Vec<u8> {
        let mut nodes: Vec<&Node> = self.graph.nodes.values().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        let mut body = Vec::new();
        body.extend_from_slice(&(nodes.len() as u32).to_le_bytes());
        for n in nodes {
            write_field(&mut body, n.id.0.as_bytes());
            write_field(&mut body, n.label.as_bytes());
            write_field(&mut body, n.kind.as_bytes());
            // Option<String>: one presence byte (1 = Some, 0 = None), then a framed field if Some.
            match &n.source {
                Some(src) => {
                    body.push(1);
                    write_field(&mut body, src.as_bytes());
                }
                None => body.push(0),
            }
        }
        body
    }

    fn encode_edges(&self) -> Vec<u8> {
        let mut edges: Vec<&Edge> = self.graph.out.values().flatten().collect();
        edges.sort_by(|a, b| {
            (a.source.cmp(&b.source))
                .then(a.target.cmp(&b.target))
                .then(a.relation.cmp(&b.relation))
        });
        let mut body = Vec::new();
        body.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for e in edges {
            write_field(&mut body, e.source.0.as_bytes());
            write_field(&mut body, e.target.0.as_bytes());
            write_field(&mut body, e.relation.as_bytes());
            body.push(confidence_tag(e.confidence));
        }
        body
    }

    fn encode_index(&self) -> Vec<u8> {
        let snap = self.index.snapshot();
        let mut body = Vec::new();
        body.extend_from_slice(&snap.next.to_le_bytes());

        // by_id: emit in sorted-by-NodeId order → deterministic.
        let mut by_id: Vec<(&NodeId, &u64)> = snap.by_id.iter().collect();
        by_id.sort_by(|a, b| a.0.cmp(b.0));
        body.extend_from_slice(&(by_id.len() as u32).to_le_bytes());
        for (id, u) in by_id {
            write_field(&mut body, id.0.as_bytes());
            body.extend_from_slice(&u.to_le_bytes());
        }

        // by_u64: emit in sorted-by-u64 order → deterministic.
        let mut by_u64: Vec<(&u64, &NodeId)> = snap.by_u64.iter().collect();
        by_u64.sort_by_key(|(u, _)| **u);
        body.extend_from_slice(&(by_u64.len() as u32).to_le_bytes());
        for (u, id) in by_u64 {
            body.extend_from_slice(&u.to_le_bytes());
            write_field(&mut body, id.0.as_bytes());
        }

        // tombstoned: sorted for determinism (order is semantically irrelevant — it's a dead-slot set).
        let mut tombs = snap.tombstoned.clone();
        tombs.sort_unstable();
        body.extend_from_slice(&(tombs.len() as u32).to_le_bytes());
        for t in tombs {
            body.extend_from_slice(&t.to_le_bytes());
        }

        // merged_into: sorted by key.
        let mut merged: Vec<(&u64, &u64)> = snap.merged_into.iter().collect();
        merged.sort_by_key(|(k, _)| **k);
        body.extend_from_slice(&(merged.len() as u32).to_le_bytes());
        for (from, into) in merged {
            body.extend_from_slice(&from.to_le_bytes());
            body.extend_from_slice(&into.to_le_bytes());
        }
        body
    }

    fn encode_provenance(&self) -> Vec<u8> {
        let mut entries: Vec<&(String, Provenance)> = self.provenance.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        let mut body = Vec::new();
        body.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (subject, p) in entries {
            write_field(&mut body, subject.as_bytes());
            write_field(&mut body, p.asserting_agent.as_bytes());
            body.push(confidence_tag(p.basis));
            write_field(&mut body, &p.parent_hash);
            write_field(&mut body, p.algorithm.as_bytes());
            body.extend_from_slice(&(p.attestations.len() as u32).to_le_bytes());
            for att in &p.attestations {
                write_field(&mut body, att.identity.as_bytes());
                write_field(&mut body, att.role.as_bytes());
                write_field(&mut body, &att.signature);
            }
        }
        body
    }
}

/// Errors from [`Artifact::from_bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// The leading 8 bytes were not the `EMGRAPH\0` magic.
    BadMagic,
    /// The format version is not one this build understands.
    UnsupportedVersion(u32),
    /// The buffer ended mid-field (a length ran past the end).
    Truncated,
    /// Bytes remained after the last section was parsed.
    TrailingBytes,
    /// A tag byte (e.g. a [`Confidence`] tag) was outside its valid range.
    BadEnumTag(u8),
    /// A framed field was not valid UTF-8 where a string was required.
    Utf8,
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArtifactError::BadMagic => write!(f, "bad magic: not an ember-graph artifact"),
            ArtifactError::UnsupportedVersion(v) => write!(f, "unsupported format version: {v}"),
            ArtifactError::Truncated => write!(f, "truncated: buffer ended mid-field"),
            ArtifactError::TrailingBytes => write!(f, "trailing bytes after final section"),
            ArtifactError::BadEnumTag(t) => write!(f, "bad enum tag: {t}"),
            ArtifactError::Utf8 => write!(f, "invalid UTF-8 in a string field"),
        }
    }
}

impl std::error::Error for ArtifactError {}

// --- framing primitives (write side) ---

/// Append a `[u32 LE len][bytes]` framed field.
fn write_field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// Append a `[u32 LE len][bytes]` framed section (same shape as a field; named for intent).
fn write_section(out: &mut Vec<u8>, body: &[u8]) {
    write_field(out, body);
}

// --- framing primitives (read side) ---

/// A strict forward cursor over the container bytes.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    /// Take exactly `n` bytes, or [`ArtifactError::Truncated`].
    fn take(&mut self, n: usize) -> Result<&'a [u8], ArtifactError> {
        let end = self.pos.checked_add(n).ok_or(ArtifactError::Truncated)?;
        if end > self.buf.len() {
            return Err(ArtifactError::Truncated);
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Read a `u32` LE.
    fn u32(&mut self) -> Result<u32, ArtifactError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a `u64` LE.
    fn u64(&mut self) -> Result<u64, ArtifactError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    /// Read one tag byte.
    fn u8(&mut self) -> Result<u8, ArtifactError> {
        Ok(self.take(1)?[0])
    }

    /// Read a `[u32 len][bytes]` framed field as raw bytes.
    fn field(&mut self) -> Result<&'a [u8], ArtifactError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    /// A framed field interpreted as a UTF-8 `String`.
    fn string(&mut self) -> Result<String, ArtifactError> {
        let b = self.field()?;
        std::str::from_utf8(b)
            .map(|s| s.to_string())
            .map_err(|_| ArtifactError::Utf8)
    }

    /// A framed section as its own bounded [`Reader`].
    fn section(&mut self) -> Result<Reader<'a>, ArtifactError> {
        Ok(Reader::new(self.field()?))
    }

    fn at_end(&self) -> bool {
        self.pos == self.buf.len()
    }
}

// --- section decoders ---

fn decode_nodes(mut r: Reader<'_>) -> Result<Vec<Node>, ArtifactError> {
    let count = r.u32()? as usize;
    let mut nodes = Vec::with_capacity(count);
    for _ in 0..count {
        let id = NodeId(r.string()?);
        let label = r.string()?;
        let kind = r.string()?;
        let source = match r.u8()? {
            0 => None,
            1 => Some(r.string()?),
            t => return Err(ArtifactError::BadEnumTag(t)),
        };
        nodes.push(Node {
            id,
            label,
            kind,
            source,
        });
    }
    if !r.at_end() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(nodes)
}

fn decode_edges(mut r: Reader<'_>) -> Result<Vec<Edge>, ArtifactError> {
    let count = r.u32()? as usize;
    let mut edges = Vec::with_capacity(count);
    for _ in 0..count {
        let source = NodeId(r.string()?);
        let target = NodeId(r.string()?);
        let relation = r.string()?;
        let confidence = confidence_from_tag(r.u8()?)?;
        edges.push(Edge {
            source,
            target,
            relation,
            confidence,
        });
    }
    if !r.at_end() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(edges)
}

fn decode_index(mut r: Reader<'_>) -> Result<NodeIndex, ArtifactError> {
    let next = r.u64()?;

    let by_id_len = r.u32()? as usize;
    let mut by_id = HashMap::with_capacity(by_id_len);
    for _ in 0..by_id_len {
        let id = NodeId(r.string()?);
        let u = r.u64()?;
        by_id.insert(id, u);
    }

    let by_u64_len = r.u32()? as usize;
    let mut by_u64 = HashMap::with_capacity(by_u64_len);
    for _ in 0..by_u64_len {
        let u = r.u64()?;
        let id = NodeId(r.string()?);
        by_u64.insert(u, id);
    }

    let tomb_len = r.u32()? as usize;
    let mut tombstoned = Vec::with_capacity(tomb_len);
    for _ in 0..tomb_len {
        tombstoned.push(r.u64()?);
    }

    let merged_len = r.u32()? as usize;
    let mut merged_into = HashMap::with_capacity(merged_len);
    for _ in 0..merged_len {
        let from = r.u64()?;
        let into = r.u64()?;
        merged_into.insert(from, into);
    }

    if !r.at_end() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(NodeIndex::from_snapshot(NodeIndexSnapshot {
        next,
        by_id,
        by_u64,
        tombstoned,
        merged_into,
    }))
}

fn decode_provenance(mut r: Reader<'_>) -> Result<Vec<(String, Provenance)>, ArtifactError> {
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let subject = r.string()?;
        let asserting_agent = r.string()?;
        let basis = confidence_from_tag(r.u8()?)?;
        let parent_hash = r.field()?.to_vec();
        let algorithm = r.string()?;
        let att_count = r.u32()? as usize;
        let mut attestations = Vec::with_capacity(att_count);
        for _ in 0..att_count {
            let identity = r.string()?;
            let role = r.string()?;
            let signature = r.field()?.to_vec();
            attestations.push(Attestation {
                identity,
                role,
                signature,
            });
        }
        out.push((
            subject,
            Provenance {
                asserting_agent,
                basis,
                parent_hash,
                algorithm,
                attestations,
            },
        ));
    }
    if !r.at_end() {
        return Err(ArtifactError::TrailingBytes);
    }
    Ok(out)
}

// --- Confidence <-> tag byte (mirrors provenance::basis_tag) ---

/// Encode a [`Confidence`] as a single tag byte.
fn confidence_tag(c: Confidence) -> u8 {
    match c {
        Confidence::Extracted => 0,
        Confidence::Inferred(InferredTier::Strong) => 1,
        Confidence::Inferred(InferredTier::Clear) => 2,
        Confidence::Inferred(InferredTier::Reasonable) => 3,
        Confidence::Inferred(InferredTier::Weak) => 4,
        Confidence::Inferred(InferredTier::Tenuous) => 5,
        Confidence::Ambiguous => 6,
    }
}

/// Decode a tag byte into a [`Confidence`], or [`ArtifactError::BadEnumTag`].
fn confidence_from_tag(t: u8) -> Result<Confidence, ArtifactError> {
    Ok(match t {
        0 => Confidence::Extracted,
        1 => Confidence::Inferred(InferredTier::Strong),
        2 => Confidence::Inferred(InferredTier::Clear),
        3 => Confidence::Inferred(InferredTier::Reasonable),
        4 => Confidence::Inferred(InferredTier::Weak),
        5 => Confidence::Inferred(InferredTier::Tenuous),
        6 => Confidence::Ambiguous,
        other => return Err(ArtifactError::BadEnumTag(other)),
    })
}

/// Human-readable name for a confidence (for the JSON export).
fn confidence_name(c: Confidence) -> &'static str {
    match c {
        Confidence::Extracted => "Extracted",
        Confidence::Inferred(InferredTier::Strong) => "Inferred(Strong)",
        Confidence::Inferred(InferredTier::Clear) => "Inferred(Clear)",
        Confidence::Inferred(InferredTier::Reasonable) => "Inferred(Reasonable)",
        Confidence::Inferred(InferredTier::Weak) => "Inferred(Weak)",
        Confidence::Inferred(InferredTier::Tenuous) => "Inferred(Tenuous)",
        Confidence::Ambiguous => "Ambiguous",
    }
}

// --- JSON helpers (hand-rolled, zero-dep) ---

/// Quote and escape a string as a JSON string literal (quotes, backslash, control chars).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Lowercase hex of a byte slice.
fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::Attestation;

    fn node(id: &str, label: &str, kind: &str, source: Option<&str>) -> Node {
        Node {
            id: NodeId(id.into()),
            label: label.into(),
            kind: kind.into(),
            source: source.map(|s| s.into()),
        }
    }

    fn edge(s: &str, t: &str, rel: &str, c: Confidence) -> Edge {
        Edge {
            source: NodeId(s.into()),
            target: NodeId(t.into()),
            relation: rel.into(),
            confidence: c,
        }
    }

    /// A non-trivial graph: nodes with/without source, edges across every confidence flavour.
    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(node("c:john", "John Doe", "contact", Some("rec:1")));
        g.add_node(node("b:acme", "Acme LLC", "business", None));
        g.add_node(node(
            "i:42",
            "Invoice 42 (unpaid)",
            "invoice",
            Some("hawksoft:42"),
        ));
        g.add_node(node("p:rug", "9x12 wool rug \"deluxe\"", "product", None));
        g.add_edge(edge("c:john", "b:acme", "owns", Confidence::Extracted));
        g.add_edge(edge("b:acme", "i:42", "owes", Confidence::Ambiguous));
        g.add_edge(edge(
            "c:john",
            "p:rug",
            "asked_about",
            Confidence::Inferred(InferredTier::Clear),
        ));
        g.add_edge(edge(
            "b:acme",
            "p:rug",
            "stocks",
            Confidence::Inferred(InferredTier::Tenuous),
        ));
        g.add_edge(edge(
            "c:john",
            "i:42",
            "disputes",
            Confidence::Inferred(InferredTier::Weak),
        ));
        g
    }

    /// An index exercised so `tombstoned` AND `merged_into` are both non-empty.
    fn sample_index() -> NodeIndex {
        let mut ix = NodeIndex::new();
        ix.intern(&NodeId("c:john".into()));
        ix.intern(&NodeId("b:acme".into()));
        ix.intern(&NodeId("i:42".into()));
        ix.intern(&NodeId("p:rug".into()));
        let dead = NodeId("x:dead".into());
        ix.intern(&dead);
        ix.tombstone(&dead); // tombstoned non-empty
                             // merge a duplicate into the survivor → merged_into + a second tombstone.
        ix.intern(&NodeId("b:acme_dup".into()));
        ix.merge(&NodeId("b:acme_dup".into()), &NodeId("b:acme".into()));
        ix
    }

    fn sample_provenance() -> Vec<(String, Provenance)> {
        let p = Provenance {
            asserting_agent: "acme:receptionist".into(),
            basis: Confidence::Inferred(InferredTier::Clear),
            parent_hash: vec![0xde, 0xad, 0xbe, 0xef],
            algorithm: "mock-v1".into(),
            attestations: vec![Attestation {
                identity: "acme:receptionist".into(),
                role: "agent-signs-action".into(),
                signature: vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef],
            }],
        };
        vec![("edge:c:john->p:rug".into(), p)]
    }

    fn sample_artifact() -> Artifact {
        let mut a = Artifact::new(sample_graph(), sample_index());
        for (subject, env) in sample_provenance() {
            a.attach_provenance(subject, env);
        }
        a
    }

    #[test]
    fn round_trip_is_faithful() {
        let a = sample_artifact();
        let bytes = a.to_bytes();
        let b = Artifact::from_bytes(&bytes).expect("round-trips");

        // Node set + each node's fields.
        assert_eq!(a.graph.node_count(), b.graph.node_count());
        for id_str in ["c:john", "b:acme", "i:42", "p:rug"] {
            let id = NodeId(id_str.into());
            assert_eq!(a.graph.node(&id), b.graph.node(&id), "node {id_str}");
        }

        // Each node's edges (relation + confidence). The artifact's canonical form sorts edges by
        // `(source, target, relation)`, so a round-tripped graph carries its per-source adjacency in
        // that canonical order; we compare as sets (sort both sides) rather than asserting the
        // original insertion order, which canonicalization deliberately discards.
        let sorted = |edges: &[Edge]| {
            let mut v = edges.to_vec();
            v.sort_by(|x, y| x.target.cmp(&y.target).then(x.relation.cmp(&y.relation)));
            v
        };
        for id_str in ["c:john", "b:acme", "i:42", "p:rug"] {
            let id = NodeId(id_str.into());
            assert_eq!(
                sorted(a.graph.neighbors(&id)),
                sorted(b.graph.neighbors(&id)),
                "edges of {id_str}"
            );
        }

        // Byte-level idempotency: re-serializing the reconstructed artifact reproduces the exact
        // input bytes (the canonical form is a fixed point — what "round-trips to_bytes exactly" means).
        assert_eq!(
            b.to_bytes(),
            bytes,
            "to_bytes is idempotent through from_bytes"
        );

        // Index: capacity / live_count / resolve of the merged-away u64 / tombstones.
        assert_eq!(a.index.capacity(), b.index.capacity());
        assert_eq!(a.index.live_count(), b.index.live_count());
        let dup = a.index.u64_of(&NodeId("b:acme_dup".into()));
        // dup was merged away, so u64_of is None on both; resolve via a known merged u64 instead.
        assert_eq!(dup, None);
        // Reconstruct the merged-away u64 from the original and check resolve survives the round-trip.
        let mut probe = NodeIndex::new();
        probe.intern(&NodeId("c:john".into()));
        probe.intern(&NodeId("b:acme".into()));
        probe.intern(&NodeId("i:42".into()));
        probe.intern(&NodeId("p:rug".into()));
        probe.intern(&NodeId("x:dead".into()));
        probe.tombstone(&NodeId("x:dead".into()));
        probe.intern(&NodeId("b:acme_dup".into()));
        let (from_u64, into_u64) =
            probe.merge(&NodeId("b:acme_dup".into()), &NodeId("b:acme".into()));
        assert_eq!(a.index.resolve(from_u64), into_u64);
        assert_eq!(
            b.index.resolve(from_u64),
            into_u64,
            "merge lineage survives"
        );

        let mut a_tombs = a.index.tombstones().to_vec();
        let mut b_tombs = b.index.tombstones().to_vec();
        a_tombs.sort_unstable();
        b_tombs.sort_unstable();
        assert_eq!(a_tombs, b_tombs, "tombstones survive");

        // Provenance table.
        assert_eq!(a.provenance, b.provenance);
    }

    #[test]
    fn output_is_deterministic_regardless_of_insertion_order() {
        let a = sample_artifact();

        // Build the SAME logical content in a different insertion order: nodes added in reverse
        // first (so each carries its real label, not a bare auto-created one), then edges in reverse.
        let mut g2 = Graph::new();
        g2.add_node(node("p:rug", "9x12 wool rug \"deluxe\"", "product", None));
        g2.add_node(node(
            "i:42",
            "Invoice 42 (unpaid)",
            "invoice",
            Some("hawksoft:42"),
        ));
        g2.add_node(node("b:acme", "Acme LLC", "business", None));
        g2.add_node(node("c:john", "John Doe", "contact", Some("rec:1")));
        g2.add_edge(edge(
            "c:john",
            "i:42",
            "disputes",
            Confidence::Inferred(InferredTier::Weak),
        ));
        g2.add_edge(edge(
            "b:acme",
            "p:rug",
            "stocks",
            Confidence::Inferred(InferredTier::Tenuous),
        ));
        g2.add_edge(edge(
            "c:john",
            "p:rug",
            "asked_about",
            Confidence::Inferred(InferredTier::Clear),
        ));
        g2.add_edge(edge("b:acme", "i:42", "owes", Confidence::Ambiguous));
        g2.add_edge(edge("c:john", "b:acme", "owns", Confidence::Extracted));

        // index built in a different intern order (but same final logical state).
        let mut ix2 = NodeIndex::new();
        ix2.intern(&NodeId("c:john".into()));
        ix2.intern(&NodeId("b:acme".into()));
        ix2.intern(&NodeId("i:42".into()));
        ix2.intern(&NodeId("p:rug".into()));
        ix2.intern(&NodeId("x:dead".into()));
        ix2.tombstone(&NodeId("x:dead".into()));
        ix2.intern(&NodeId("b:acme_dup".into()));
        ix2.merge(&NodeId("b:acme_dup".into()), &NodeId("b:acme".into()));

        let mut a2 = Artifact::new(g2, ix2);
        for (subject, env) in sample_provenance() {
            a2.attach_provenance(subject, env);
        }

        assert_eq!(
            a.to_bytes(),
            a2.to_bytes(),
            "byte-identical for equal logical content"
        );
    }

    /// Parse, expecting failure, and return the error (`Artifact` has no `Debug`/`PartialEq`, so we
    /// can't `assert_eq!` on the whole `Result` — extract the error and compare that instead).
    fn err(bytes: &[u8]) -> ArtifactError {
        Artifact::from_bytes(bytes)
            .err()
            .expect("expected a parse error")
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_artifact().to_bytes();
        bytes[0] = b'X';
        assert_eq!(err(&bytes), ArtifactError::BadMagic);
    }

    #[test]
    fn rejects_unknown_version() {
        let mut bytes = sample_artifact().to_bytes();
        // version is the 4 LE bytes right after the 8-byte magic; bump it to 2.
        bytes[8] = 2;
        assert_eq!(err(&bytes), ArtifactError::UnsupportedVersion(2));
    }

    #[test]
    fn rejects_truncation() {
        let bytes = sample_artifact().to_bytes();
        let truncated = &bytes[..bytes.len() - 5];
        assert_eq!(err(truncated), ArtifactError::Truncated);
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut bytes = sample_artifact().to_bytes();
        bytes.push(0x00); // one extra byte after the last section
        assert_eq!(err(&bytes), ArtifactError::TrailingBytes);
    }

    #[test]
    fn rejects_bad_confidence_tag() {
        // Build a minimal artifact with exactly one edge so we can find its confidence tag byte.
        let mut g = Graph::new();
        g.add_edge(edge("a", "b", "rel", Confidence::Extracted));
        let a = Artifact::new(g, NodeIndex::new());
        let mut bytes = a.to_bytes();
        // The edge's confidence tag is the very last byte of the edges section. Rather than compute
        // its offset, find the lone valid tag 0 that is the edge confidence by parsing structurally:
        // simplest robust approach — corrupt every byte to 99 and confirm at least the tag path is hit
        // is fragile, so instead locate it deterministically.
        let tag_pos = locate_edge_confidence_tag(&bytes);
        bytes[tag_pos] = 99;
        assert_eq!(err(&bytes), ArtifactError::BadEnumTag(99));
    }

    /// Walk the container to the single edge's confidence tag byte and return its absolute index.
    fn locate_edge_confidence_tag(bytes: &[u8]) -> usize {
        // magic(8) + version(4) + nodes_section_len(4) + nodes_body + edges_section_len(4) ...
        let mut pos = 8 + 4;
        let nodes_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4 + nodes_len; // skip nodes section
        let _edges_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4; // now at edges body start
                  // edges body: u32 count, then [src field][tgt field][rel field][tag byte]
        let _count = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        for _ in 0..3 {
            let flen = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + flen;
        }
        pos // the confidence tag byte
    }

    #[test]
    fn rejects_bad_utf8() {
        // Hand-build a tiny container with one node whose id field has invalid UTF-8.
        let mut body = Vec::new();
        body.extend_from_slice(&1u32.to_le_bytes()); // node count = 1
        write_field(&mut body, &[0xff, 0xfe]); // id: invalid UTF-8
        write_field(&mut body, b"label");
        write_field(&mut body, b"kind");
        body.push(0); // source = None

        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        write_section(&mut bytes, &body); // nodes
        write_section(&mut bytes, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u32.to_le_bytes());
            b
        }); // edges: 0
        write_section(&mut bytes, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u64.to_le_bytes()); // next
            for _ in 0..4 {
                b.extend_from_slice(&0u32.to_le_bytes()); // by_id, by_u64, tombs, merged counts
            }
            b
        }); // index
        write_section(&mut bytes, &{
            let mut b = Vec::new();
            b.extend_from_slice(&0u32.to_le_bytes());
            b
        }); // provenance: 0

        assert_eq!(err(&bytes), ArtifactError::Utf8);
    }

    #[test]
    fn to_json_contains_expected_substrings() {
        let json = sample_artifact().to_json();
        assert!(json.contains("John Doe"), "a node label: {json}");
        assert!(json.contains("asked_about"), "a relation: {json}");
        assert!(json.contains("disputes"), "another relation");
        // signature 01 23 45 67 89 ab cd ef → hex prefix.
        assert!(json.contains("0123456789abcdef"), "hex signature: {json}");
        assert!(json.contains("deadbeef"), "hex parent_hash");
        // proper escaping of the embedded quote in the rug label.
        assert!(json.contains("\\\"deluxe\\\""), "escaped quotes: {json}");
    }

    #[test]
    fn empty_provenance_round_trips() {
        let a = Artifact::new(sample_graph(), sample_index());
        assert!(a.provenance.is_empty());
        let bytes = a.to_bytes();
        let b = Artifact::from_bytes(&bytes).unwrap();
        assert!(b.provenance.is_empty());
        assert_eq!(a.graph.node_count(), b.graph.node_count());
    }

    #[test]
    fn error_display_is_informative() {
        assert_eq!(
            ArtifactError::UnsupportedVersion(7).to_string(),
            "unsupported format version: 7"
        );
        assert_eq!(
            ArtifactError::BadEnumTag(99).to_string(),
            "bad enum tag: 99"
        );
    }
}
