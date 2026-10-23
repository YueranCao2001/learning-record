"""Explicit allowlisted publication to the agreed private repo; no token output."""
import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = "YueranCao2001/qwen3-0.6b-zh-extraction-lora"
ALLOWED = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
           "tokenizer.json", "special_tokens_map.json", "vocab.json", "merges.txt",
           "chat_template.jinja", "added_tokens.json", "README.md", "LICENSE",
           "manifest.json", "environment.json", "evaluation.json"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-card", default="model-card.md")
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    results = ROOT / "results" / args.run
    artifact = ROOT / "artifacts" / args.run
    if args.arm:
        results = results / args.arm
        artifact = artifact / args.arm
    assert Path(args.model_card).name == args.model_card
    assert Path(args.revision).name == args.revision
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert summary["base_unchanged"] and summary["adapter_reload_exact"]
    assert summary["reload_generation_tokens_equal"] and summary["steps"] == 6
    shutil.copyfile(ROOT / args.model_card, artifact / "README.md")
    base = Path("E:/Evren/Evren_work/.hf-cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    shutil.copyfile(base / "LICENSE", artifact / "LICENSE")
    for name in ("manifest.json", "environment.json"):
        shutil.copyfile(results / name, artifact / name)
    write_json(artifact / "evaluation.json", {
        "summary": summary,
        "baseline": json.loads((results / "baseline.json").read_text(encoding="utf-8")),
        "after": json.loads((results / "after.json").read_text(encoding="utf-8"))})
    files = sorted(p.name for p in artifact.iterdir())
    assert set(files) <= ALLOWED and (artifact / "adapter_model.safetensors").is_file()
    hashes = {name: hashlib.sha256((artifact / name).read_bytes()).hexdigest() for name in files}
    print(json.dumps({"files": files, "repo": REPO, "revision": args.revision}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    api = HfApi()
    assert api.whoami()["name"] == "YueranCao2001"
    assert api.model_info(REPO).private is True
    main_before = api.model_info(REPO, revision="main").sha
    if args.revision != "main":
        api.create_branch(repo_id=REPO, branch=args.revision, exist_ok=True)
    commit = api.upload_folder(repo_id=REPO, repo_type="model", folder_path=artifact,
                               allow_patterns=files, revision=args.revision,
                               commit_message="Add six-step LoRA diagnostic adapter and evidence")
    report = {"repo": REPO, "commit": commit.oid, "private": api.model_info(REPO).private,
              "revision": args.revision,
              "main_before": main_before, "main_after": api.model_info(REPO, revision="main").sha,
              "publisher_huggingface_hub": version("huggingface_hub"),
              "files_sha256": hashes, "download_hashes_equal": False,
              "remote_model_reload_verified": False}
    write_json(results / "upload.json", report)
    if args.revision != "main":
        assert report["main_before"] == report["main_after"], "main changed during branch publication"
    snapshot = Path(snapshot_download(REPO, revision=commit.oid, allow_patterns=files,
                    cache_dir="E:/Evren/Evren_work/.hf-verify-cache"))
    assert all(hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == value
               for name, value in hashes.items())
    assert report["private"] is True
    report.update(download_hashes_equal=True, downloaded_snapshot=str(snapshot))
    write_json(results / "upload.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
