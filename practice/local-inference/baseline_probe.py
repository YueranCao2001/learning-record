import argparse
import json
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from stream_probe import EXPECTED, MODEL, OPTIONS, PROMPT, probe


LABELS = {
    "short": ("Amber", "Cedar", "Dawn", "Fern", "Harbor", "Ivory"),
    "long": ("Jade", "Lemon", "Maple", "North", "Opal", "Pebble"),
}
PARAGRAPH = (
    "A field notebook records the weather, the route, the time of departure, and the "
    "condition of each trail. Each entry separates observed facts from estimates. "
    "A second reader checks the directions before using them."
)
PADDING = "\n".join(f"{index:02d}. {PARAGRAPH}" for index in range(1, 17))


def payload_for(length, label):
    task = PROMPT if length == "short" else (
        f"Ignore this background passage:\n<background>\n{PADDING}\n</background>\n\n{PROMPT}"
    )
    return {
        "model": MODEL, "think": False, "options": OPTIONS,
        "messages": [{"role": "user", "content": f"{label}\n\n{task}"}],
    }


def median(rows, field):
    return round(statistics.median(row[field] for row in rows), 3)


def summarize(rows):
    return {
        "count": len(rows),
        "first_text_median_ms": median(rows, "first_text_ms"),
        "complete_median_ms": median(rows, "complete_ms"),
        "cached_fraction_median": median(rows, "cached_fraction"),
        "prompt_eval_median_ms": round(statistics.median(
            row["server"]["prompt_eval_duration"] / 1e6 for row in rows
        ), 3),
        "decode_median_tokens_per_second": median(rows, "decode_tokens_per_second"),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure a serial Ollama baseline by prompt and cache state.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    url = f"{base_url}/api/chat"
    warmup_payload = payload_for("short", "Warmup")
    print("Running model warm-up", flush=True)
    warmup = probe(opener, url, warmup_payload, stream=True)

    rows = []
    for index in range(len(LABELS["short"])):
        lengths = ("short", "long") if index % 2 == 0 else ("long", "short")
        for length in lengths:
            label = LABELS[length][index]
            payload = payload_for(length, label)
            pair = []
            for condition in ("changed_prefix", "repeated_prefix"):
                row = probe(opener, url, payload, stream=True)
                counters = row["server"]
                cached_fraction = counters["prompt_eval_cached_count"] / counters["prompt_eval_count"]
                if condition == "changed_prefix" and cached_fraction > 0.25:
                    raise ValueError(f"Expected low cache reuse for {label}, got {cached_fraction:.1%}")
                if condition == "repeated_prefix" and cached_fraction < 0.90:
                    raise ValueError(f"Expected high cache reuse for {label}, got {cached_fraction:.1%}")
                pair.append({
                    "length": length, "label": label, "condition": condition, **row,
                    "cached_fraction": round(cached_fraction, 6),
                    "decode_tokens_per_second": round(counters["eval_count"] * 1e9 / counters["eval_duration"], 3),
                })
            if pair[0]["server"]["prompt_eval_count"] != pair[1]["server"]["prompt_eval_count"]:
                raise ValueError(f"Prompt token counts differ for {label}")
            rows.extend(pair)
            print(f"Completed {length} pair {index + 1}: {label}", flush=True)

    summary = {
        length: {
            condition: summarize([row for row in rows if row["length"] == length and row["condition"] == condition])
            for condition in ("changed_prefix", "repeated_prefix")
        }
        for length in LABELS
    }
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": url, "ollama_version": version, "model": MODEL, "options": OPTIONS,
        "expected_reply": EXPECTED, "labels": LABELS,
        "prompt_templates": {
            length: payload_for(length, "{label}")["messages"][0]["content"] for length in LABELS
        },
        "warmup": {"request": warmup_payload, **warmup}, "requests": rows, "summary": summary,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
