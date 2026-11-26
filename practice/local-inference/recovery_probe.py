import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from admission_probe import OUTCOMES, WAIT_BUDGET_MS, WAIT_LIMIT, WORKERS, run_round
from arrival_probe import SHAPE_BURST_SIZE
from cancellation_probe import dedicated_ollama
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies
from queue_probe import nearest_rank
from stream_probe import MODEL


PHASES = ("steady", "burst", "recovery")
LOW_WAIT_MS = 50


def arrival_trace(recovery_requests):
    return (
        [(ms, "steady") for ms in range(0, 6000, 1000)]
        + [(ms, "burst") for ms in range(6000, 12001, 2000) for _ in range(SHAPE_BURST_SIZE)]
        + [(12500 + index * 1000, "recovery") for index in range(recovery_requests)]
    )


def summarize_phase(rows):
    outcomes = Counter(row["outcome"] for row in rows)
    good = [row for row in rows if row["outcome"] == "completed"]
    return {
        "offered": len(rows),
        "outcomes": {outcome: outcomes[outcome] for outcome in OUTCOMES},
        "submit_to_http_p90_ms": nearest_rank((row["http_start_ms"] - row["submitted_ms"] for row in good), 0.9),
        "completed_p90_ms": nearest_rank((row["terminal_ms"] - row["submitted_ms"] for row in good), 0.9),
    }


def recovery_timing(rows):
    recovery = [row for row in rows if row["phase"] == "recovery"]
    burst = [row for row in rows if row["phase"] == "burst"]
    transition = recovery[0]["submitted_ms"]
    queued_at_transition = [row for row in burst if row["queue_exit_ms"] is not None
                            and row["queue_enter_ms"] <= transition < row["queue_exit_ms"]]
    low_wait_suffix = []
    for row in reversed(recovery):
        if row["outcome"] != "completed" or row["http_start_ms"] - row["submitted_ms"] > LOW_WAIT_MS:
            break
        low_wait_suffix.append(row)
    return {
        "first_recovery_submit_ms": transition,
        "burst_waiting_at_transition": len(queued_at_transition),
        "burst_queue_clear_after_ms": max((row["queue_exit_ms"] - transition for row in queued_at_transition), default=0),
        "burst_settled_after_ms": max(0, max(row["terminal_ms"] for row in burst) - transition),
        "low_wait_suffix_count": len(low_wait_suffix),
        "low_wait_suffix_start_after_ms": low_wait_suffix[-1]["submitted_ms"] - transition if low_wait_suffix else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare client recovery after a burst of requests.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-requests", type=int, choices=range(4, 41), metavar="4-40", default=8,
                        help="Number of 1 Hz arrivals in the recovery phase (default: 8).")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/api/chat"
    trace = arrival_trace(args.recovery_requests)
    count = len(trace)
    with dedicated_ollama(args.port, WORKERS) as (_, _, version, digest, configuration):
        rounds = []
        for repetition, order in enumerate((("baseline", "bounded"), ("bounded", "baseline")), start=1):
            for policy in order:
                run = run_round(url, repetition, policy, 600 + len(rounds) * count, arrival_trace=trace)
                run["recovery"] = recovery_timing(run["requests"])
                rounds.append(run)
                print(f"Repetition {repetition}, {policy}: {run['recovery']}", flush=True)

    summary = {}
    for policy in ("baseline", "bounded"):
        rows = [row for run in rounds if run["policy"] == policy for row in run["requests"]]
        summary[policy] = {phase: summarize_phase([row for row in rows if row["phase"] == phase]) for phase in PHASES}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": digest, "server_configuration": configuration,
        "client_in_flight_limit": WORKERS, "options": GENERATION_OPTIONS,
        "bounded_waiting_limit": WAIT_LIMIT, "bounded_wait_budget_ms": WAIT_BUDGET_MS,
        "recovery_requests_per_round": args.recovery_requests,
        "low_wait_threshold_ms": LOW_WAIT_MS, "expected_reply": LONG_REPLY,
        "actual_replies": deduplicate_replies(rounds), "rounds": rounds, "summary_by_phase": summary,
    }
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
