"""Append evaluation documents to the experimental HF branch, never model files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent
REPO = "YueranCao2001/qwen3-0.6b-zh-extraction-lora"
BRANCH = "boundary-v2-20261023"
TESTED = "e3b24d1115c21cf69feaa5e6e9cf85a98f502c84"
FILES = {"manifest.json", "review.json", "protocol.json", "environment.json",
         "base-extraction.json", "candidate-extraction.json", "base-retention.json",
         "candidate-retention.json", "annotation-rules-v1.md"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kind", choices=("audit", "minimal", "resume"), default="audit")
    parser.add_argument("--expected-head", default=TESTED)
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    review_file = "comparison.json" if args.kind == "resume" else "review.json"
    review = json.loads((output / review_file).read_text(encoding="utf-8"))
    if args.kind != "resume":
        assert review["training_steps"] == 0 and review["human_review"] == "pending"
    if args.kind == "resume":
        assert review["full_resume_exact"] and review["optimizer_reset_diverges"] and review["four_distinct_processes"]
        assert review["executed_optimizer_steps"] == 10 and not review["quality_evaluated"]
        files = {"manifest.json", "protocol.json", "environment.json", "comparison.json"}
        files |= {phase + "/" + name for phase in ("continuous", "prefix", "resume", "optimizer-reset")
                  for name in ("history.json", "state.json")}
    elif args.kind == "audit":
        assert review["old_candidate_tokens_reproduce"] and review["total_generation_calls"] == 52
        files = FILES
    else:
        assert all(review["anchor_tokens_reproduce"].values()) and review["total_generation_calls"] == 12
        assert review["source_groups"] == 1
        files = FILES - {"base-retention.json", "candidate-retention.json"}
    bundle = ROOT / "artifacts" / args.run / "audit-documents"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in files:
        source = ROOT / name if name.endswith(".md") else output / name
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, bundle / name)
    assert {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} == files
    hashes = {name: digest(bundle / name) for name in sorted(files)}
    prefix = ("engineering/" if args.kind == "resume" else "evaluations/") + args.run
    print(json.dumps({"branch": BRANCH, "prefix": prefix, "files": sorted(files), "weight_upload": False}), flush=True)
    if not args.execute:
        return
    api = HfApi()
    assert api.whoami()["name"] == "YueranCao2001"
    assert api.model_info(REPO).private is True
    main_before = api.model_info(REPO, revision="main").sha
    assert api.model_info(REPO, revision=BRANCH).sha == args.expected_head, "branch head changed; inspect before writing"
    commit = api.upload_folder(repo_id=REPO, folder_path=bundle, path_in_repo=prefix,
               revision=BRANCH, allow_patterns=sorted(files),
               commit_message="Add read-only diagnostic evidence; leave weights unchanged")
    report = {"repo": REPO, "branch": BRANCH, "commit": commit.oid,
              "tested_adapter_revision": None if args.kind == "resume" else TESTED, "reference_adapter_revision": TESTED,
              "path_in_repo": prefix, "files_sha256": hashes, "weight_upload": False,
              "main_before": main_before, "main_after": api.model_info(REPO, revision="main").sha,
              "private": api.model_info(REPO).private, "download_hashes_equal": False,
              "adapter_weights_unchanged": False}
    write = lambda: (output / "hf-audit-upload.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write()
    for name, expected in hashes.items():
        downloaded = hf_hub_download(REPO, filename=prefix + "/" + name, revision=commit.oid,
                        cache_dir="E:/Evren/Evren_work/.hf-verify-cache")
        assert digest(downloaded) == expected
    adapter = hf_hub_download(REPO, filename="adapter_model.safetensors", revision=commit.oid,
                             cache_dir="E:/Evren/Evren_work/.hf-verify-cache")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_adapter = manifest.get("published_reference_adapter_sha256", manifest.get("adapter_sha256"))
    assert digest(adapter) == expected_adapter
    assert report["main_before"] == report["main_after"] and report["private"] is True
    report.update(download_hashes_equal=True, adapter_weights_unchanged=True, adapter_sha256=expected_adapter)
    write()
    print(json.dumps({k: report[k] for k in ("commit", "private", "download_hashes_equal", "adapter_weights_unchanged", "main_after")}), flush=True)


if __name__ == "__main__":
    main()
