import argparse
import json
import statistics
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition
from time import perf_counter

from arrival_probe import EQUAL_WINDOW_SECONDS, SHAPE_BURST_SIZE, SHAPE_RATE, WORKERS
from cancellation_probe import dedicated_ollama
from concurrency_probe import local_opener, peak_in_flight
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies
from queue_probe import LONG_OUTPUT_PAYLOAD, nearest_rank
from stream_probe import MODEL, probe


WAIT_LIMIT = 1
WAIT_BUDGET_MS = 1000
BUDGET_SWEEP_MS = (1000, 1400, 1600)
ORDER_CHECK_BUDGET_MS = BUDGET_SWEEP_MS[-1]
REQUEST_COUNT = int(EQUAL_WINDOW_SECONDS * SHAPE_RATE)
BREAKDOWN_REQUEST_COUNT = REQUEST_COUNT // 2
OUTCOMES = ("completed", "deadline_before_http", "queue_full", "request_error")


class AdmissionExpired(Exception):
    pass


def execute_request(url, request, origin):
    row = {**request, "http_start_ms": None}

    def before_http(at):
        start = (at - origin) * 1000
        if row["deadline_ms"] is not None and start >= row["deadline_ms"]:
            raise AdmissionExpired
        row["http_start_ms"] = start

    try:
        payload = LONG_OUTPUT_PAYLOAD(row["length"], row["request_id"])
        row.update(probe(local_opener(), url, payload, stream=True, expected=LONG_REPLY, before_http=before_http))
        row["outcome"] = "completed"
    except AdmissionExpired:
        row["outcome"] = "deadline_before_http"
    except (OSError, ValueError) as error:
        row.update(outcome="request_error", error=f"{type(error).__name__}: {error}")
    row["terminal_ms"] = (perf_counter() - origin) * 1000
    return row


def policy_limits(policy):
    limit = WAIT_LIMIT if policy in ("queue_only", "bounded") else None
    budget_ms = WAIT_BUDGET_MS if policy in ("deadline_only", "bounded") else None
    return limit, budget_ms


def run_round(url, repetition, policy, label_offset, request_count=REQUEST_COUNT, *, limits=None, length_order="short_first"):
    wait_limit, budget_ms = policy_limits(policy) if limits is None else limits
    schedule = [1000 * (i // SHAPE_BURST_SIZE * SHAPE_BURST_SIZE) / SHAPE_RATE for i in range(request_count)]
    pending = deque()
    active = set()
    finished = []
    changed = Condition()
    origin = perf_counter()
    submitted = max_pending = 0

    def notify_completion(_):
        with changed:
            changed.notify()

    def finish_before_http(request, outcome, now):
        finished.append({**request, "http_start_ms": None, "terminal_ms": now, "outcome": outcome})

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        def grant(request):
            future = executor.submit(execute_request, url, request, origin)
            active.add(future)
            future.add_done_callback(notify_completion)

        with changed:
            while len(finished) < request_count:
                done = {future for future in active if future.done()}
                active.difference_update(done)
                finished.extend(future.result() for future in done)

                while pending:
                    deadline = pending[0]["deadline_ms"]
                    now = (perf_counter() - origin) * 1000
                    if deadline is not None and now >= deadline:
                        finish_before_http(pending.popleft(), "deadline_before_http", now)
                    elif len(active) < WORKERS:
                        grant(pending.popleft())
                    else:
                        break

                now = (perf_counter() - origin) * 1000
                if submitted < request_count and schedule[submitted] <= now:
                    batch = submitted // SHAPE_BURST_SIZE
                    starts_long = length_order == "long_first" or (
                        length_order == "alternating" and (batch + repetition - 1) % 2 == 1
                    )
                    lengths = ("long", "short") if starts_long else ("short", "long")
                    request = {
                        "request_id": f"{label_offset + submitted:03d}", "planned_ms": schedule[submitted],
                        "length": lengths[submitted % 2], "submitted_ms": now,
                        "deadline_ms": now + budget_ms if budget_ms is not None else None,
                    }
                    submitted += 1
                    if len(active) < WORKERS:
                        grant(request)
                    elif wait_limit is not None and len(pending) >= wait_limit:
                        finish_before_http(request, "queue_full", now)
                    else:
                        pending.append(request)
                        max_pending = max(max_pending, len(pending))
                    continue

                if len(finished) == request_count:
                    break
                wake_times = []
                if submitted < request_count:
                    wake_times.append(schedule[submitted])
                if budget_ms is not None and pending:
                    wake_times.append(pending[0]["deadline_ms"])
                if any(future.done() for future in active):
                    continue
                wait_seconds = max(0, (min(wake_times) - now) / 1000) if wake_times else None
                changed.wait(timeout=wait_seconds)

    finished.sort(key=lambda row: row["request_id"])
    http_intervals = [{"start_ms": row["http_start_ms"], "end_ms": row["terminal_ms"]}
                      for row in finished if row["http_start_ms"] is not None]
    end_ms = max(row["terminal_ms"] for row in finished)
    return {
        "repetition": repetition, "policy": policy, "peak_waiting": max_pending,
        "peak_http_in_flight": peak_in_flight(http_intervals),
        "drain_ms": end_ms - finished[-1]["submitted_ms"], "requests": finished,
    }


def summarize(rounds):
    rows = [row for run in rounds for row in run["requests"]]
    outcomes = Counter(row["outcome"] for row in rows)
    good = [row for row in rows if row["outcome"] == "completed"]
    return {
        "offered": len(rows), "outcomes": {outcome: outcomes[outcome] for outcome in OUTCOMES},
        "completed_share": len(good) / len(rows),
        "completed_p90_ms": nearest_rank((row["terminal_ms"] - row["submitted_ms"] for row in good), 0.9),
        "median_drain_ms": statistics.median(run["drain_ms"] for run in rounds),
    }


def summarize_by_length(rounds):
    rows = [row for run in rounds for row in run["requests"]]
    return {
        length: {"offered": sum(row["length"] == length for row in rows),
                 "completed": sum(row["length"] == length and row["outcome"] == "completed" for row in rows)}
        for length in ("short", "long")
    }


def main():
    parser = argparse.ArgumentParser(description="Compare finite client admission with the same loopback burst trace.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--policy-breakdown", action="store_true", help="Isolate queue capacity and waiting budget in two 20-request repetitions.")
    mode.add_argument("--budget-sweep", action="store_true", help="Try several waiting budgets with one waiting slot.")
    mode.add_argument("--order-check", action="store_true", help="Compare short-first, long-first and alternating batches at a fixed budget.")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/api/chat"
    policies = ("baseline", "queue_only", "deadline_only", "bounded") if args.policy_breakdown else ("baseline", "bounded")
    limits = {name: policy_limits(name) for name in policies}
    if args.budget_sweep:
        limits = {f"budget_{budget}": (WAIT_LIMIT, budget) for budget in BUDGET_SWEEP_MS}
        policies = tuple(limits)
    elif args.order_check:
        policies = ("short_first", "long_first", "alternating")
        limits = {policy: (WAIT_LIMIT, ORDER_CHECK_BUDGET_MS) for policy in policies}
    request_count = BREAKDOWN_REQUEST_COUNT if args.policy_breakdown or args.budget_sweep or args.order_check else REQUEST_COUNT
    with dedicated_ollama(args.port, WORKERS) as (_, _, version, digest, configuration):
        rounds = []
        for repetition, order in enumerate((policies, policies[::-1]), start=1):
            for policy in order:
                run = run_round(url, repetition, policy, 600 + len(rounds) * request_count, request_count,
                                limits=limits[policy], length_order=policy if args.order_check else "short_first")
                rounds.append(run)
                print(f"Repetition {repetition}, {policy}: {summarize([run])['outcomes']}", flush=True)

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": digest, "server_configuration": configuration,
        "client_in_flight_limit": WORKERS, "options": GENERATION_OPTIONS,
        "requests_per_round": request_count, "burst_size": SHAPE_BURST_SIZE,
        "burst_interval_ms": 1000 * SHAPE_BURST_SIZE / SHAPE_RATE,
        "expected_reply": LONG_REPLY, "actual_replies": deduplicate_replies(rounds), "rounds": rounds,
        "summary": {policy: summarize([run for run in rounds if run["policy"] == policy]) for policy in policies},
    }
    if not args.budget_sweep and not args.order_check:
        report.update(bounded_waiting_limit=WAIT_LIMIT, bounded_wait_budget_ms=WAIT_BUDGET_MS)
    if args.policy_breakdown or args.budget_sweep or args.order_check:
        report["policies"] = {
            policy: dict(zip(("waiting_limit", "wait_budget_ms"), limits[policy])) for policy in policies
        }
    if args.order_check:
        for policy in policies:
            report["summary"][policy]["by_length"] = summarize_by_length([run for run in rounds if run["policy"] == policy])
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
