import argparse
import json
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from baseline_probe import payload_for
from stream_probe import EXPECTED, MODEL, OPTIONS, probe


LENGTHS = ("short", "long", "short", "long")
ROUNDS = 4


def local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def run_request(url, length, label, start, build_payload, expected):
    opener = local_opener()
    payload = build_payload(length, label)
    sent = perf_counter()
    try:
        result = probe(opener, url, payload, stream=True, expected=expected)
    except (OSError, ValueError) as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    ended = perf_counter()
    return {"length": length, "label": label, "start_ms": (sent - start) * 1000,
            "end_ms": (ended - start) * 1000, **result}


def peak_in_flight(rows):
    edges = [(row["start_ms"], 1) for row in rows] + [(row["end_ms"], -1) for row in rows]
    active = peak = 0
    for _, change in sorted(edges):
        active += change
        peak = max(peak, active)
    return peak


def run_round(url, number, workers, offset, *, build_payload=payload_for, expected=EXPECTED):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        started = perf_counter()
        tasks = [executor.submit(run_request, url, length, f"{offset + i:03d}", started, build_payload, expected)
                 for i, length in enumerate(LENGTHS)]
        rows = [task.result() for task in tasks]
        wall_seconds = perf_counter() - started
    good = [row for row in rows if "error" not in row]
    return {
        "round": number, "concurrency": workers, "peak_in_flight": peak_in_flight(rows),
        "wall_ms": wall_seconds * 1000, "successes": len(good), "failures": len(rows) - len(good),
        "requests_per_second": len(good) / wall_seconds,
        "output_tokens_per_second": sum(row["server"]["eval_count"] for row in good) / wall_seconds,
        "requests": rows,
    }


def summarize(rounds):
    good = [row for run in rounds for row in run["requests"] if "error" not in row]
    return {
        "successes": len(good), "failures": sum(run["failures"] for run in rounds),
        "first_text_median_ms": statistics.median(row["first_text_ms"] for row in good) if good else None,
        "complete_median_ms": statistics.median(row["complete_ms"] for row in good) if good else None,
        "round_requests_per_second_median": statistics.median(run["requests_per_second"] for run in rounds),
        "round_output_tokens_per_second_median": statistics.median(run["output_tokens_per_second"] for run in rounds),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare one versus two in-flight Ollama requests.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    opener = local_opener()
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    url = f"{base_url}/api/chat"
    warmup = run_round(url, 0, 2, 900)
    if warmup["failures"]:
        raise RuntimeError(f"Warm-up failed: {warmup['requests']}")

    rounds = []
    for number in range(1, ROUNDS + 1):
        order = (1, 2) if number % 2 else (2, 1)
        for workers in order:
            offset = 100 + len(rounds) * len(LENGTHS)
            run = run_round(url, number, workers, offset)
            rounds.append(run)
            print(f"Round {number}, concurrency {workers}: {run['successes']}/{len(LENGTHS)} passed", flush=True)

    summary = {workers: summarize([run for run in rounds if run["concurrency"] == workers]) for workers in (1, 2)}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": url, "ollama_version": version, "model": MODEL, "options": OPTIONS,
        "expected_reply": EXPECTED, "warmup": warmup, "rounds": rounds, "summary": summary,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
