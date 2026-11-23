import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

from cancellation_probe import dedicated_ollama
from concurrency_probe import peak_in_flight, run_request
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies
from queue_probe import LONG_OUTPUT_PAYLOAD, nearest_rank
from stream_probe import MODEL


RATES = (1.0, 2.0)
REQUESTS_PER_ROUND = 10
WORKERS = 2


def run_round(url, repetition, rate, label_offset):
    submissions = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        origin = perf_counter()
        for index in range(REQUESTS_PER_ROUND):
            planned = index / rate
            sleep(max(0, origin + planned - perf_counter()))
            submitted_ms = (perf_counter() - origin) * 1000
            length = "short" if index % 2 == 0 else "long"
            task = executor.submit(run_request, url, length, f"{label_offset + index:03d}", origin,
                                   LONG_OUTPUT_PAYLOAD, LONG_REPLY)
            submissions.append((planned * 1000, submitted_ms, task))
        rows = [{"planned_ms": planned, "submitted_ms": submitted, **task.result()}
                for planned, submitted, task in submissions]

    for index, row in enumerate(rows):
        submitted = row["submitted_ms"]
        row["schedule_delay_ms"] = submitted - row["planned_ms"]
        row["client_wait_ms"] = row["start_ms"] - submitted
        row["arrival_to_complete_ms"] = row["end_ms"] - submitted
        row["already_waiting_at_arrival"] = sum(
            earlier["submitted_ms"] <= submitted < earlier["start_ms"] for earlier in rows[:index]
        )

    final_arrival = rows[-1]["submitted_ms"]
    final_completion = max(row["end_ms"] for row in rows)
    good = [row for row in rows if "error" not in row]
    return {
        "repetition": repetition, "target_arrivals_per_second": rate,
        "actual_arrivals_per_second": (len(rows) - 1) * 1000 / (final_arrival - rows[0]["submitted_ms"]),
        "peak_in_flight": peak_in_flight(rows), "successes": len(good), "failures": len(rows) - len(good),
        "max_already_waiting_at_arrival": max(row["already_waiting_at_arrival"] for row in rows),
        "unfinished_after_last_arrival": sum(row["end_ms"] > final_arrival for row in rows),
        "drain_ms": final_completion - final_arrival,
        "completion_throughput_per_second": len(good) * 1000 / (final_completion - rows[0]["submitted_ms"]),
        "requests": rows,
    }


def summarize(rounds):
    rows = [row for run in rounds for row in run["requests"]]
    return {
        "successes": sum(run["successes"] for run in rounds), "failures": sum(run["failures"] for run in rounds),
        "client_wait_p90_ms": nearest_rank((row["client_wait_ms"] for row in rows), 0.9),
        "arrival_to_complete_p90_ms": nearest_rank((row["arrival_to_complete_ms"] for row in rows if "error" not in row), 0.9),
        "max_already_waiting_at_arrival": max(run["max_already_waiting_at_arrival"] for run in rounds),
        "unfinished_after_last_arrival": [run["unfinished_after_last_arrival"] for run in rounds],
        "median_drain_ms": statistics.median(run["drain_ms"] for run in rounds),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay bounded open-loop arrivals on a loopback Ollama server.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/api/chat"

    with dedicated_ollama(args.port, WORKERS) as (_, _, version, digest, configuration):
        rounds = []
        for repetition, rates in enumerate((RATES, RATES[::-1]), start=1):
            for rate in rates:
                run = run_round(url, repetition, rate, 300 + len(rounds) * REQUESTS_PER_ROUND)
                rounds.append(run)
                print(f"Repetition {repetition}, {rate:g} arrivals/s: {run['successes']}/{REQUESTS_PER_ROUND} passed", flush=True)

    summary = {rate: summarize([run for run in rounds if run["target_arrivals_per_second"] == rate]) for rate in RATES}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": digest, "server_configuration": configuration,
        "client_in_flight_limit": WORKERS, "requests_per_round": REQUESTS_PER_ROUND, "options": GENERATION_OPTIONS,
        "expected_reply": LONG_REPLY, "actual_replies": deduplicate_replies(rounds),
        "rounds": rounds, "summary": summary,
    }
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
