import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Event
from time import perf_counter

from cancellation_probe import dedicated_ollama, open_chat
from concurrency_probe import local_opener
from deadline_probe import timed_request
from output_length_probe import LONG_REPLY, payload_for
from pre_first_cancel_probe import run_trial
from queue_probe import LONG_OUTPUT_PAYLOAD
from stream_probe import EXPECTED, MODEL, probe, read_stream, validate_reply


ADMISSION_BUDGET_MS = 100


def verify_stream(port, log):
    begin = len(log.events)
    url = f"http://127.0.0.1:{port}/api/chat"
    result = probe(local_opener(), url, payload_for("short", "101", output_length="short"), stream=True)
    if result["text_chunks"] < 2 or result["first_text_ms"] > result["complete_ms"]:
        raise ValueError(f"Expected a multi-chunk reply with valid timings: {result}")
    return {"response": result, "server_log_events": log.after_releases(begin, 1)}


def verify_admission_timeout(port, log, origin):
    begin = len(log.events)
    gate = BoundedSemaphore(1)
    first_text = Event()
    holder_times = {}

    def hold_slot():
        def on_first_text():
            holder_times["first_text_ms"] = (perf_counter() - origin) * 1000
            first_text.set()

        with gate:
            connection, response = open_chat(port, LONG_OUTPUT_PAYLOAD("long", "201"))
            with closing(connection), response:
                final, text, *_ = read_stream(response, perf_counter(), on_first_text)
                validate_reply(final, text, MODEL, LONG_REPLY)
            holder_times["completed_ms"] = (perf_counter() - origin) * 1000

    with ThreadPoolExecutor(max_workers=1) as executor:
        holder = executor.submit(hold_slot)
        if not first_text.wait(timeout=4):
            if holder.done():
                holder.result()
            raise TimeoutError("The slot holder did not produce first text")
        submitted = perf_counter()
        waiting = timed_request(port, "short", "202", submitted, submitted + ADMISSION_BUDGET_MS / 1000, gate)
        decision_ms = (perf_counter() - origin) * 1000
        holder.result()
    if (waiting["outcome"] != "deadline_before_http" or waiting["http_start_ms"] is not None
            or waiting["http_status"] is not None or decision_ms >= holder_times["completed_ms"]):
        raise ValueError(f"Expected admission timeout while the slot holder was still active: {waiting}")
    events = log.after_releases(begin, 1)
    starts = [e for e in events if e["kind"] == "start"]
    if len(starts) != 1 or starts[0]["tokens"] != 889 or any(e["kind"] == "cancel" for e in events):
        raise ValueError(f"Only the successful slot holder should appear in runner logs: {events}")
    return {"budget_ms": ADMISSION_BUDGET_MS, "waiting": waiting, "decision_ms": decision_ms,
            "holder": holder_times, "holder_reply_validated": True, "server_log_events": events}


def verify_cancellation(port, log, origin, mode, offset):
    result = run_trial(port, log, origin, 1, mode, offset)
    if result["candidate_entered_slot"] != (mode != "queued_disconnect") or not result["follow_reply_validated"]:
        raise ValueError(f"Unexpected candidate or follow-up slot result for {mode}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the local Ollama inference acceptance checks.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": f"http://127.0.0.1:{args.port}/api/chat",
        "model": MODEL, "status": "failed", "cases": [],
    }
    current_case = "startup"
    try:
        with dedicated_ollama(args.port, 1) as (log, origin, version, digest, configuration):
            report.update(ollama_version=version, model_digest=digest, server_configuration=configuration)
            current_case = "normal_stream"
            details = verify_stream(args.port, log)
            report["cases"].append({"name": current_case, "status": "passed", "evidence": details})

            current_case = "admission_timeout"
            details = verify_admission_timeout(args.port, log, origin)
            report["cases"].append({"name": current_case, "status": "passed", "evidence": details})

            for current_case, offset in (("queued_disconnect", 300), ("active_disconnect", 400), ("queued_complete", 500)):
                details = verify_cancellation(args.port, log, origin, current_case, offset)
                report["cases"].append({"name": current_case, "status": "passed", "evidence": details})
            current_case = "shutdown"
        report["status"] = "passed"
    except Exception as error:
        report["cases"].append({"name": current_case, "status": "failed", "error": f"{type(error).__name__}: {error}"})

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = sum(case["status"] == "passed" for case in report["cases"])
    print(f"Local inference acceptance {report['status']}: {passed}/5 checks passed; {args.output}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
