import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import admission_probe as admission
import arrival_probe as arrival
import recovery_probe as recovery
from cancellation_probe import dedicated_ollama
from concurrency_probe import peak_in_flight
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies
from stream_probe import MODEL


ARCHIVE_COUNTS = {
    "2026-11-24-arrival-shape": 160,
    "2026-11-24-capacity": 128,
    "2026-11-24-capacity-equal-window": 168,
    "2026-11-24-capacity-midpoint": 96,
    "2026-11-25-bounded-admission": 160,
    "2026-11-25-admission-breakdown": 160,
    "2026-11-25-admission-budget-sweep": 120,
    "2026-11-25-admission-input-order": 120,
    "2026-11-26-admission-recovery": 120,
    "2026-11-26-admission-recovery-extended": 200,
}


def require(condition, context):
    if not condition:
        raise ValueError(context)


def verify_rows(report):
    require(report["model"] == MODEL and report["options"] == GENERATION_OPTIONS, "model or options mismatch")
    config = report["server_configuration"]
    require(config["slots"] == report["client_in_flight_limit"] == 2 and config["context_per_slot"] == 2048,
            "slot configuration mismatch")
    require(report["expected_reply"] == LONG_REPLY, "reply contract mismatch")
    reply_digest = hashlib.sha256(LONG_REPLY.encode("utf-8")).hexdigest()
    require(report["actual_replies"].get(reply_digest, "").strip() == LONG_REPLY, "missing expected reply")

    seen = set()
    totals = Counter()
    for run in report["rounds"]:
        intervals = []
        counts = Counter()
        rows = run["requests"]
        for row in rows:
            identifier = row.get("request_id", row.get("label"))
            status = row.get("outcome", "request_error" if "error" in row else "completed")
            start = row.get("http_start_ms", row.get("start_ms"))
            end = row.get("terminal_ms", row.get("end_ms"))
            submit = row["submitted_ms"]
            deadline = row.get("deadline_ms")
            require(identifier and identifier not in seen, f"duplicate or missing request ID: {identifier}")
            seen.add(identifier)
            require(row["planned_ms"] - 0.01 <= submit <= end, f"invalid request clock: {identifier}")
            if status == "completed":
                require(start is not None and submit <= start <= end, f"invalid HTTP interval: {identifier}")
                require(deadline is None or start < deadline, f"HTTP started at or after deadline: {identifier}")
                require((row["http_status"], row["done_reason"], row["reply_sha256"]) == (200, "stop", reply_digest),
                        f"invalid completed reply: {identifier}")
                tokens = row["server"]
                require(tokens["prompt_eval_count"] == {"short": 190, "long": 889}[row["length"]]
                        and tokens["eval_count"] == 117, f"token counts changed: {identifier}")
                intervals.append({"start_ms": start, "end_ms": end})
            elif status == "deadline_before_http":
                require(start is None and deadline is not None and end >= deadline, f"invalid deadline: {identifier}")
            elif status == "queue_full":
                require(start is None and end == submit, f"invalid immediate rejection: {identifier}")
            else:
                raise ValueError(f"unexpected request outcome: {identifier} ({status})")
            counts[status] += 1
        peak = run.get("peak_http_in_flight", run.get("peak_in_flight"))
        require(peak_in_flight(intervals) == peak <= 2, "HTTP concurrency mismatch")
        if "peak_waiting" in run:
            limits = report.get("policies", {}).get(run["policy"])
            limit, budget = admission.policy_limits(run["policy"]) if limits is None else (
                limits["waiting_limit"], limits["wait_budget_ms"])
            require(limit is None or run["peak_waiting"] <= limit, "waiting capacity exceeded")
            require(all(row["deadline_ms"] == (row["submitted_ms"] + budget if budget is not None else None)
                        for row in rows), "deadline budget mismatch")
        if "recovery" in run:
            queue_intervals = [{"start_ms": row["queue_enter_ms"], "end_ms": row["queue_exit_ms"]}
                               for row in rows if row["queue_enter_ms"] is not None]
            require(peak_in_flight(queue_intervals) == run["peak_waiting"], "waiting event peak mismatch")
            require(recovery.recovery_timing(rows) == run["recovery"], "recovery metrics mismatch")
        totals.update(counts)
    return dict(totals)


def verify_archives(results):
    verified = {}
    for name, count in ARCHIVE_COUNTS.items():
        path = results / f"{name}.json"
        payload = path.read_bytes()
        report = json.loads(payload)
        try:
            outcomes = verify_rows(report)
            require(sum(outcomes.values()) == count, "archive count mismatch")
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        verified[path.name] = {"sha256": hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest(),
                               "runs": len(report["rounds"]), "outcomes": outcomes}
    return verified


def run_live(port):
    url = f"http://127.0.0.1:{port}/api/chat"
    with dedicated_ollama(port, arrival.WORKERS) as (_, _, version, digest, config):
        paced = arrival.run_round(url, 1, 1.25, 810, 12)
        print("Completed the 12-request paced replay", flush=True)
        burst = [admission.run_round(url, 1, policy, offset, 8)
                 for policy, offset in (("baseline", 830), ("bounded", 840))]
        print("Completed the 16-request burst replay", flush=True)
        phased = []
        for policy, offset in (("baseline", 850), ("bounded", 920)):
            run = admission.run_round(url, 1, policy, offset, arrival_trace=recovery.arrival_trace(8))
            run["recovery"] = recovery.recovery_timing(run["requests"])
            phased.append(run)
            print(f"Completed the 30-request recovery replay ({policy})", flush=True)
    rounds = [paced, *burst, *phased]
    report = {
        "url": url, "ollama_version": version, "model": MODEL, "model_digest": digest,
        "server_configuration": config, "client_in_flight_limit": arrival.WORKERS, "options": GENERATION_OPTIONS,
        "expected_reply": LONG_REPLY, "actual_replies": deduplicate_replies(rounds), "rounds": rounds,
    }
    report["outcomes"] = verify_rows(report)
    paced_metrics = arrival.summarize([paced])
    report["paced_summary"] = paced_metrics
    report["checks"] = {
        "paced_complete_with_low_wait": paced["successes"] == 12 and paced_metrics["client_wait_p90_ms"] <= recovery.LOW_WAIT_MS,
        "baseline_burst_completed": all(row["outcome"] == "completed" for row in burst[0]["requests"]),
        "bounded_burst_outcomes": [row["outcome"] for row in burst[1]["requests"]]
        == ["completed", "completed", "deadline_before_http", "queue_full"] * 2,
        "baseline_recovery_completed": all(row["outcome"] == "completed" for row in phased[0]["requests"]),
        "bounded_recovery_low_wait_suffix": phased[1]["recovery"]["low_wait_suffix_count"] >= 4,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit archived local inference traces and optionally replay short workloads.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--live", action="store_true", help="Also run 88 requests on a fresh loopback instance.")
    args = parser.parse_args()
    archives = verify_archives(Path(__file__).with_name("results"))
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "archive_hash_encoding": "LF",
              "archive": archives}
    print(f"Verified {sum(sum(entry['outcomes'].values()) for entry in archives.values())} archived requests", flush=True)
    if args.live:
        report["live"] = run_live(args.port)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.live and not all(report["live"]["checks"].values()):
        raise SystemExit("Live replay did not pass every check. See the report for details.")


if __name__ == "__main__":
    main()
