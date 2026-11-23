import argparse
import json
import math
import statistics
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


def summarize_batch_arrival(rounds):
    attempts = [row for run in rounds for row in run["requests"]]
    successful = [row for row in attempts if "error" not in row]
    first_text_times = [row["start_ms"] + row["first_text_ms"] for row in successful]
    return {
        "submit_to_http_p90_ms": nearest_rank((row["start_ms"] for row in attempts), 0.9),
        "submit_to_first_text_p90_ms": nearest_rank(first_text_times, 0.9),
        "submit_to_complete_p90_ms": nearest_rank((row["end_ms"] for row in successful), 0.9),
        "batch_wall_median_ms": statistics.median(run["wall_ms"] for run in rounds),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two and four in-flight requests against two Ollama slots.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-arrival", action="store_true",
                        help="Release four requests together and time them from that shared arrival.")
    args = parser.parse_args()
    lengths = LENGTHS if args.batch_arrival else REQUEST_LENGTHS
    count = 8 if args.batch_arrival else ROUNDS

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
        for number in range(1, count + 1):
            order = (2, 4) if number % 2 else (4, 2)
            for workers in order:
                offset = 400 + len(rounds) * len(lengths)
                run = run_round(url, number, workers, offset, build_payload=LONG_OUTPUT_PAYLOAD,
                                expected=LONG_REPLY, lengths=lengths)
                rounds.append(run)
                print(f"Round {number}, concurrency {workers}: {run['successes']}/{len(lengths)} passed", flush=True)

    summary = {}
    if rounds:
        for workers in (2, 4):
            group = [run for run in rounds if run["concurrency"] == workers]
            summary[workers] = summarize_queue(group)
            if args.batch_arrival:
                summary[workers].update(summarize_batch_arrival(group))
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": model["digest"],
        "options": GENERATION_OPTIONS, "expected_reply": LONG_REPLY,
        "actual_replies": deduplicate_replies([warmup, *rounds]),
        "prompt_templates": {length: LONG_OUTPUT_PAYLOAD(length, "{label}")["messages"][0]["content"]
                             for length in ("short", "long")},
        "warmup": warmup, "rounds": rounds, "summary": summary,
    }
    if args.batch_arrival:
        report["time_origin"] = "Shared batch arrival immediately before all four requests are submitted to the client executor."
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, separators=(",", ":"))
        output.write("\n")
    print(json.dumps(summary, indent=2))
    if warmup["failures"]:
        raise SystemExit("Warm-up reply validation failed; diagnostics were saved")


if __name__ == "__main__":
    main()
