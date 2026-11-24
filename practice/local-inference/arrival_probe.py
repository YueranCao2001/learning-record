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
CAPACITY_RATES = (1.25, 1.5)
CAPACITY_REQUESTS_PER_ROUND = 32
MIDPOINT_RATE = 1.375
MIDPOINT_REQUESTS_PER_ROUND = 48
EQUAL_WINDOW_SECONDS = 32
EQUAL_WINDOW_RATES = (1.25, MIDPOINT_RATE)
SHAPE_RATE = EQUAL_WINDOW_RATES[0]
SHAPE_BURST_SIZE = 4
WORKERS = 2


def run_round(url, repetition, rate, label_offset, request_count=REQUESTS_PER_ROUND, burst_size=1):
    submissions = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        origin = perf_counter()
        for index in range(request_count):
            planned = (index // burst_size * burst_size) / rate
            if index % burst_size == 0:
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


def summarize_halves(rows):
    midpoint = len(rows) // 2
    result = {}
    for name, half in (("first", rows[:midpoint]), ("second", rows[midpoint:])):
        result[name] = {
            "client_wait_median_ms": statistics.median(row["client_wait_ms"] for row in half),
            "client_wait_p90_ms": nearest_rank((row["client_wait_ms"] for row in half), 0.9),
            "max_already_waiting_at_arrival": max(row["already_waiting_at_arrival"] for row in half),
        }
    return result


def run_arrival_shapes(url):
    rounds = []
    shapes = {"paced": 1, "burst": SHAPE_BURST_SIZE}
    order = tuple(shapes)
    count = int(EQUAL_WINDOW_SECONDS * SHAPE_RATE)
    boundary_ms = EQUAL_WINDOW_SECONDS * 1000
    for repetition, shape_order in enumerate((order, order[::-1]), start=1):
        for shape in shape_order:
            run = run_round(url, repetition, SHAPE_RATE, 500 + len(rounds) * count, count, shapes[shape])
            offered = [row for row in run["requests"] if row["submitted_ms"] < boundary_ms]
            finished = [row for row in offered if row["end_ms"] < boundary_ms]
            run.update(
                arrival_shape=shape, burst_size=shapes[shape],
                arrivals_in_window=len(offered), arrivals_per_second_in_window=len(offered) / EQUAL_WINDOW_SECONDS,
                successes_in_window=sum("error" not in row for row in finished),
                unfinished_at_window_end=len(offered) - len(finished),
            )
            rounds.append(run)
            print(f"Repetition {repetition}, {shape}: {run['successes']}/{count} passed", flush=True)
    return rounds


def main():
    parser = argparse.ArgumentParser(description="Replay bounded open-loop arrivals on a loopback Ollama server.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capacity", action="store_true", help="Run longer windows at 1.25 and 1.5 arrivals per second.")
    mode.add_argument("--capacity-midpoint", action="store_true", help="Run two 48-request windows at 1.375 arrivals per second.")
    mode.add_argument("--capacity-equal-window", action="store_true", help="Compare 1.25 and 1.375 arrivals per second over equal 32-second windows.")
    mode.add_argument("--arrival-shape", action="store_true", help="Compare paced arrivals and bursts of four at the same 32-second average rate.")
    args = parser.parse_args()
    rates, request_count = RATES, REQUESTS_PER_ROUND
    if args.capacity:
        rates, request_count = CAPACITY_RATES, CAPACITY_REQUESTS_PER_ROUND
    elif args.capacity_midpoint:
        rates, request_count = (MIDPOINT_RATE,), MIDPOINT_REQUESTS_PER_ROUND
    elif args.capacity_equal_window:
        rates = EQUAL_WINDOW_RATES
    url = f"http://127.0.0.1:{args.port}/api/chat"

    counts = {rate: int(rate * EQUAL_WINDOW_SECONDS) if args.capacity_equal_window else request_count for rate in rates}
    with dedicated_ollama(args.port, WORKERS) as (_, _, version, digest, configuration):
        if args.arrival_shape:
            rounds = run_arrival_shapes(url)
        else:
            rounds = []
            label_offset = 300
            for repetition, rate_order in enumerate((rates, rates[::-1]), start=1):
                for rate in rate_order:
                    count = counts[rate]
                    run = run_round(url, repetition, rate, label_offset, count)
                    label_offset += count
                    if args.capacity or args.capacity_midpoint or args.capacity_equal_window:
                        run["halves"] = summarize_halves(run["requests"])
                    rounds.append(run)
                    print(f"Repetition {repetition}, {rate:g} arrivals/s: {run['successes']}/{count} passed", flush=True)

    if args.arrival_shape:
        summary = {shape: summarize([run for run in rounds if run["arrival_shape"] == shape])
                   for shape in ("paced", "burst")}
    else:
        summary = {rate: summarize([run for run in rounds if run["target_arrivals_per_second"] == rate]) for rate in rates}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": digest, "server_configuration": configuration,
        "client_in_flight_limit": WORKERS, "options": GENERATION_OPTIONS,
        "expected_reply": LONG_REPLY, "actual_replies": deduplicate_replies(rounds),
        "rounds": rounds, "summary": summary,
    }
    if args.arrival_shape:
        report["planned_arrival_window_seconds"] = EQUAL_WINDOW_SECONDS
        report["requests_per_round"] = int(EQUAL_WINDOW_SECONDS * SHAPE_RATE)
    elif args.capacity_equal_window:
        report["planned_arrival_window_seconds"] = EQUAL_WINDOW_SECONDS
        report["requests_per_round_by_rate"] = counts
    else:
        report["requests_per_round"] = request_count
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
