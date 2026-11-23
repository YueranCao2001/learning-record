import argparse
import json
import math
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from concurrency_probe import LENGTHS, local_opener, run_round, summarize
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies, payload_for
from stream_probe import MODEL


ROUNDS = 6
REQUEST_LENGTHS = LENGTHS * 2
LONG_OUTPUT_PAYLOAD = partial(payload_for, output_length="long")


def nearest_rank(values, quantile):
    values = sorted(values)
    return values[math.ceil(len(values) * quantile) - 1] if values else None


def summarize_queue(rounds):
    rows = [row for run in rounds for row in run["requests"] if "error" not in row]
    return {
        **summarize(rounds),
        "first_text_p90_ms": nearest_rank((row["first_text_ms"] for row in rows), 0.9),
        "complete_p90_ms": nearest_rank((row["complete_ms"] for row in rows), 0.9),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two and four in-flight requests against two Ollama slots.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    opener = local_opener()
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    with opener.open(f"{base_url}/api/tags", timeout=10) as response:
        model = next(item for item in json.load(response)["models"] if item["name"] == MODEL)
    url = f"{base_url}/api/chat"
    warmup = run_round(url, 0, 2, 900, build_payload=LONG_OUTPUT_PAYLOAD, expected=LONG_REPLY)
    print(f"Warm-up: {warmup['successes']}/{len(LENGTHS)} passed", flush=True)

    rounds = []
    if not warmup["failures"]:
        for number in range(1, ROUNDS + 1):
            order = (2, 4) if number % 2 else (4, 2)
            for workers in order:
                offset = 400 + len(rounds) * len(REQUEST_LENGTHS)
                run = run_round(url, number, workers, offset, build_payload=LONG_OUTPUT_PAYLOAD,
                                expected=LONG_REPLY, lengths=REQUEST_LENGTHS)
                rounds.append(run)
                print(f"Round {number}, concurrency {workers}: {run['successes']}/{len(REQUEST_LENGTHS)} passed", flush=True)

    summary = {
        workers: summarize_queue([run for run in rounds if run["concurrency"] == workers])
        for workers in (2, 4)
    } if rounds else {}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": model["digest"],
        "options": GENERATION_OPTIONS, "expected_reply": LONG_REPLY,
        "actual_replies": deduplicate_replies([warmup, *rounds]),
        "prompt_templates": {length: LONG_OUTPUT_PAYLOAD(length, "{label}")["messages"][0]["content"]
                             for length in ("short", "long")},
        "warmup": warmup, "rounds": rounds, "summary": summary,
    }
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, separators=(",", ":"))
        output.write("\n")
    print(json.dumps(summary, indent=2))
    if warmup["failures"]:
        raise SystemExit("Warm-up reply validation failed; diagnostics were saved")


if __name__ == "__main__":
    main()
