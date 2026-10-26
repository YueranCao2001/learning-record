import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock
from checkpoint_container import MAGIC, MAX_HEADER, make_header, encode_header, decode_stream, write_atomic, ContainerRejected

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/2026-10-26-atomic-checkpoint-v1"


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


class ContainerTests(unittest.TestCase):
    def setUp(self):
        self.payload = b"controlled test payload"
        self.binding = {"schema_version": 1, "identity": {"data": "fixed"},
                        "checkpoint_sha256": hashlib.sha256(self.payload).hexdigest()}
        self.header = make_header(self.payload, self.binding, "generation")
        self.blob = encode_header(self.header) + self.payload

    def test_valid_callback(self):
        callback = Mock(return_value="decoded")
        header, result = decode_stream(io.BytesIO(self.blob), {"data": "fixed"}, callback)
        self.assertEqual(header["generation"], "generation")
        self.assertEqual(result, "decoded")
        callback.assert_called_once_with(self.payload)

    def test_truncation_rejected(self):
        callback = Mock()
        with self.assertRaisesRegex(ContainerRejected, "payload_size"):
            decode_stream(io.BytesIO(self.blob[:-3]), {"data": "fixed"}, callback)
        callback.assert_not_called()

    def test_payload_corruption_rejected(self):
        callback = Mock()
        with self.assertRaisesRegex(ContainerRejected, "payload_sha256"):
            decode_stream(io.BytesIO(self.blob[:-1] + b"!"), {"data": "fixed"}, callback)
        callback.assert_not_called()

    def test_header_limit_rejected(self):
        callback = Mock()
        with self.assertRaisesRegex(ContainerRejected, "header_limit"):
            decode_stream(io.BytesIO(MAGIC + struct.pack(">Q", MAX_HEADER + 1)), {}, callback)
        callback.assert_not_called()

    def test_identity_rejected(self):
        callback = Mock()
        with self.assertRaisesRegex(ContainerRejected, "identity_mismatch"):
            decode_stream(io.BytesIO(self.blob), {"data": "different"}, callback)
        callback.assert_not_called()

    def test_failed_publication_keeps_old_slot(self):
        artifact_root = (ROOT / "artifacts").resolve()
        temporary = tempfile.TemporaryDirectory(dir=artifact_root)
        folder = Path(temporary.name).resolve()
        self.assertTrue(folder.is_relative_to(artifact_root))
        try:
            source = folder / "source.pt"
            source.write_bytes(self.payload)
            target = folder / "slot.ckpt"
            write_atomic(target, source, self.binding, "old")
            before = target.read_bytes()

            def fail(stage):
                raise OSError("injected write failure")

            with self.assertRaises(OSError):
                write_atomic(target, source, self.binding, "new", hook=fail)
            self.assertEqual(target.read_bytes(), before)
        finally:
            temporary.cleanup()


class AtomicSavedTests(unittest.TestCase):
    def test_provenance(self):
        manifest = read("manifest.json")
        for name, expected in manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_controlled_writer_exits(self):
        cases = {r["case"]: r for r in read("writer-cases.json")}
        self.assertEqual(len(cases), 4)
        for name in ("mid-payload", "before-replace"):
            self.assertTrue(cases[name]["old_bytes_preserved"] and cases[name]["readable"])
            self.assertEqual(cases[name]["generation"], "old-generation")
            self.assertEqual(cases[name]["pending_file_count"], 1)
        for name in ("after-replace", "normal"):
            self.assertTrue(cases[name]["readable"])
            self.assertEqual(cases[name]["generation"], "new-generation")

    def test_invalid_inputs_do_not_deserialize(self):
        cases = read("corruption-cases.json")
        self.assertEqual(len(cases), 5)
        self.assertTrue(all(r["rejected"] and not r["deserializer_called"] and r["optimizer_updates"] == 0 for r in cases))

    def test_gpu_and_hf_json_only(self):
        gpu = read("gpu-resume.json")
        self.assertTrue(gpu["exact"] and all(gpu["checks"].values()))
        self.assertEqual(gpu["adapter_max_abs_diff"], 0)
        self.assertEqual(gpu["optimizer_updates"], 2)
        self.assertFalse(read("comparison.json")["real_power_failure_tested"])
        upload = read("hf-audit-upload.json")
        self.assertEqual(len(upload["files_sha256"]), 9)
        self.assertTrue(all(name.endswith(".json") for name in upload["files_sha256"]))
        self.assertTrue(upload["download_hashes_equal"] and upload["adapter_weights_unchanged"] and upload["private"])
        self.assertFalse(upload["weight_upload"])
        self.assertEqual(upload["main_before"], upload["main_after"])


if __name__ == "__main__":
    unittest.main()
