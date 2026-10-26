"""Fail-closed pre-deserialization checks for locally produced resume states."""
import hashlib
import json
from pathlib import Path


class GuardRejected(ValueError):
    pass


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_identity(expected, actual):
    if not isinstance(expected, dict) or not isinstance(actual, dict) or set(expected) != set(actual):
        raise GuardRejected("identity_fields")
    mismatches = sorted(key for key in expected if expected[key] != actual[key])
    if mismatches:
        raise GuardRejected("identity_mismatch:" + ",".join(mismatches))


def guarded_load(envelope, checkpoint, runtime_identity, deserialize):
    if envelope.get("schema_version") != 1 or not isinstance(envelope.get("identity"), dict):
        raise GuardRejected("invalid_envelope")
    if file_digest(checkpoint) != envelope.get("checkpoint_sha256"):
        raise GuardRejected("checkpoint_sha256")
    validate_identity(envelope["identity"], runtime_identity)
    # No payload deserialization occurs until all file/identity checks above pass.
    state = deserialize(checkpoint)
    required = {"adapter", "optimizer", "scheduler", "rng", "history", "parameter_names", "completed_steps", "protocol_sha256"}
    if not isinstance(state, dict) or not required <= set(state):
        raise GuardRejected("checkpoint_schema")
    if state["completed_steps"] != 2 or len(state["history"]) != 2:
        raise GuardRejected("checkpoint_position")
    if state["protocol_sha256"] != runtime_identity["training_protocol_sha256"]:
        raise GuardRejected("checkpoint_protocol")
    if canonical_digest(state["parameter_names"]) != envelope["parameter_order_sha256"]:
        raise GuardRejected("parameter_order")
    return state
