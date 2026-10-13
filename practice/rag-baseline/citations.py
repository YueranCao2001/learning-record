"""Exact provenance checks, not semantic entailment or source authentication."""
import hashlib
from dataclasses import dataclass

from retrieval import rank


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    source_id: str
    version: str
    start: int
    end: int
    text: str


def load_snapshots(data):
    if data["schema_version"] != 1 or data["synthetic"] is not True:
        raise ValueError("expected synthetic snapshot schema v1")
    snapshots = {}
    for source in data["sources"]:
        key = (source["source_id"], source["version"])
        if key in snapshots:
            raise ValueError("duplicate source version")
        if not all(isinstance(x, str) and x for x in key) or not isinstance(source["text"], str):
            raise ValueError("invalid snapshot")
        snapshots[key] = source["text"]
    return snapshots


def make_evidence(source_id, version, text, start=0, end=None):
    end = len(text) if end is None else end
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError("invalid evidence range")
    body = text[start:end]
    identity = repr((source_id, version, start, end, body)).encode("utf-8")
    return Evidence(hashlib.sha256(identity).hexdigest(), source_id, version, start, end, body)


def assemble(docs, query, user, analyzer, method, snapshots, versions, k=2):
    # The application owns versions and snapshots; model output cannot set them.
    hits = rank(docs, query, user, analyzer, method, k)
    by_id = {d.id: d for d in docs}
    evidence = []
    for hit in hits:
        doc = by_id[hit["id"]]
        version = versions.get(doc.id)
        source = snapshots.get((doc.id, version))
        if source is None or source != doc.text:
            raise ValueError("retrieval text differs from bound snapshot")
        evidence.append(make_evidence(doc.id, version, source))
    return tuple(evidence)


def check_quote(citation, evidence, snapshots, allowed_sources):
    # allowed_sources must come from the current trusted authorization layer.
    indexed = {c.chunk_id: c for c in evidence}
    if len(indexed) != len(evidence):
        raise ValueError("duplicate evidence ID")
    chunk = indexed.get(citation["chunk_id"])
    if chunk is None:
        raise ValueError("citation not in supplied evidence")
    if chunk.source_id not in allowed_sources:
        raise ValueError("source no longer authorized")
    if (citation["source_id"], citation["version"]) != (chunk.source_id, chunk.version):
        raise ValueError("citation source or version mismatch")
    source = snapshots.get((chunk.source_id, chunk.version))
    if source is None:
        raise ValueError("bound source version missing")
    expected = make_evidence(chunk.source_id, chunk.version, source, chunk.start, chunk.end)
    if expected != chunk:
        raise ValueError("snapshot or evidence mismatch")
    start, end = citation["start"], citation["end"]
    if type(start) is not int or type(end) is not int:
        raise ValueError("invalid citation offsets")
    if not chunk.start <= start < end <= chunk.end:
        raise ValueError("citation outside supplied range")
    if not isinstance(citation["quote"], str) or source[start:end] != citation["quote"]:
        raise ValueError("quote differs from source")
    return True


def full_quote(evidence):
    return {"chunk_id": evidence.chunk_id, "source_id": evidence.source_id,
            "version": evidence.version, "start": evidence.start,
            "end": evidence.end, "quote": evidence.text}
