"""Load the freshly downloaded pinned Hub adapter and compare diagnostic tokens."""
import argparse
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, set_seed
from contract import load_pilot
from train_smoke import ROOT, load_base, generate, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", choices=("control", "candidate"))
    args = parser.parse_args()
    assert Path(args.run).name == args.run
    output = ROOT / "results" / args.run
    if args.arm:
        output = output / args.arm
    report = json.loads((output / "upload.json").read_text(encoding="utf-8"))
    assert report["private"] and report["download_hashes_equal"]
    snapshot = Path(report["downloaded_snapshot"])
    assert snapshot.name == report["commit"]
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = PeftModel.from_pretrained(load_base(), snapshot, local_files_only=True).eval()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    data_path = (ROOT / manifest.get("data_file", "data/pilot-v1.json")).resolve()
    assert data_path.is_relative_to(ROOT.resolve())
    actual = generate(model, tokenizer, load_pilot(data_path))
    expected = json.loads((output / "after.json").read_text(encoding="utf-8"))
    save_json(output / "remote-reloaded.json", actual)
    assert [r["token_ids"] for r in actual] == [r["token_ids"] for r in expected]
    report["remote_model_reload_verified"] = True
    save_json(output / "upload.json", report)
    print(json.dumps({"commit": report["commit"], "remote_model_reload_verified": True}), flush=True)


if __name__ == "__main__":
    main()
