"""Bounded, checksummed single-file containers; controlled single-writer publication."""
import hashlib
import json
import os
from pathlib import Path
import struct
import uuid
from resume_guard import file_digest, validate_identity, GuardRejected

MAGIC = b"LRCKPT1\n"
MAX_HEADER = 1024 * 1024
MAX_PAYLOAD = 64 * 1024 * 1024


class ContainerRejected(ValueError):
    pass


def make_header(payload, binding, generation):
    sha = hashlib.sha256(payload).hexdigest()
    if binding.get("checkpoint_sha256") != sha:
        raise ContainerRejected("source_payload_binding")
    return {"container_version": 1, "generation": generation, "binding": binding,
            "payload_bytes": len(payload), "payload_sha256": sha}


def encode_header(header):
    data = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_HEADER:
        raise ContainerRejected("header_limit")
    return MAGIC + struct.pack(">Q", len(data)) + data


def write_atomic(target, payload_path, binding, generation, fault=None, hook=None):
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_digest(payload_path) != binding.get("checkpoint_sha256"):
        raise ContainerRejected("source_file_binding")
    payload = Path(payload_path).read_bytes()
    if not 0 < len(payload) <= MAX_PAYLOAD:
        raise ContainerRejected("payload_limit")
    header = make_header(payload, binding, generation)
    temporary = target.parent / (".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(encode_header(header))
        if fault == "mid-payload":
            stream.write(payload[:len(payload) // 2])
            stream.flush()
            os.fsync(stream.fileno())
            os._exit(91)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if hook:
        hook("before-replace")
    if fault == "before-replace":
        os._exit(92)
    os.replace(temporary, target)
    if fault == "after-replace":
        os._exit(93)
    return header


def decode_stream(stream, runtime_identity, deserialize):
    if stream.read(len(MAGIC)) != MAGIC:
        raise ContainerRejected("magic")
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ContainerRejected("header_length_missing")
    size = struct.unpack(">Q", length_bytes)[0]
    if not 0 < size <= MAX_HEADER:
        raise ContainerRejected("header_limit")
    encoded = stream.read(size)
    if len(encoded) != size:
        raise ContainerRejected("header_truncated")
    try:
        header = json.loads(encoded)
    except (ValueError, UnicodeError) as exc:
        raise ContainerRejected("header_json") from exc
    if not isinstance(header, dict) or header.get("container_version") != 1:
        raise ContainerRejected("container_schema")
    count = header.get("payload_bytes")
    if type(count) is not int or not 0 < count <= MAX_PAYLOAD:
        raise ContainerRejected("payload_limit")
    payload = stream.read(count + 1)
    if len(payload) != count:
        raise ContainerRejected("payload_size")
    binding = header.get("binding")
    if not isinstance(binding, dict) or binding.get("schema_version") != 1:
        raise ContainerRejected("binding_schema")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != header.get("payload_sha256") or actual != binding.get("checkpoint_sha256"):
        raise ContainerRejected("payload_sha256")
    try:
        validate_identity(binding.get("identity"), runtime_identity)
    except GuardRejected as exc:
        raise ContainerRejected(str(exc)) from exc
    # Deserialize exactly the bytes validated from this open handle, not a reopened path.
    return header, deserialize(payload)


def load_container(path, runtime_identity, deserialize):
    with Path(path).open("rb") as stream:
        return decode_stream(stream, runtime_identity, deserialize)
