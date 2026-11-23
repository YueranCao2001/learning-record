import argparse
import http.client
import json
import select
import socket
import statistics
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import perf_counter

from baseline_probe import PADDING
from cancellation_probe import dedicated_ollama, open_chat
from output_length_probe import LONG_REPLY, payload_for
from queue_probe import LONG_OUTPUT_PAYLOAD
from stream_probe import EXPECTED, MODEL, read_stream, validate_reply


MODES = ("queued_disconnect", "active_disconnect", "queued_complete")
QUEUE_OBSERVATION_SECONDS = 0.08


def candidate_payload(label):
    payload = LONG_OUTPUT_PAYLOAD("short", label)
    message = payload["messages"][0]
    heading, task = message["content"].split("\n\n", 1)
    message["content"] = f"{heading}\n\nIgnore this background:\n{PADDING}\n{PADDING}\n\n{task}"
    return payload


def run_trial(port, log, origin, repetition, mode, offset):
    begin = len(log.events)
    times = {}
    holder_first = Event()
    disconnect = mode != "queued_complete"

    def mark(name):
        times[name] = (perf_counter() - origin) * 1000

    def consume(role, response, expected, signal=None):
        def on_first_text():
            mark(f"{role}_first_text")
            if signal is not None:
                signal.set()

        final, text, *_ = read_stream(response, perf_counter(), on_first_text)
        validate_reply(final, text, MODEL, expected)
        mark(f"{role}_completed")

    def request(role, payload, expected, signal=None):
        connection, response = open_chat(port, payload, lambda: mark(f"{role}_posted"))
        with closing(connection), response:
            consume(role, response, expected, signal)

    def finish_candidate(connection):
        with connection.getresponse() as response:
            if response.status != 200:
                raise ValueError(f"Unexpected candidate HTTP status: {response.status}")
            consume("candidate", response, LONG_REPLY)

    with ThreadPoolExecutor(max_workers=3) as executor:
        holder = None
        if mode != "active_disconnect":
            holder = executor.submit(request, "holder", LONG_OUTPUT_PAYLOAD("long", f"{offset:03d}"),
                                     LONG_REPLY, holder_first)
            if not holder_first.wait(timeout=4):
                if holder.done():
                    holder.result()
                raise TimeoutError("Slot holder did not produce first text")

        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as connection:
            payload = candidate_payload(f"{offset + 1:03d}")
            connection.request("POST", "/api/chat", json.dumps({**payload, "stream": True}).encode("utf-8"),
                               {"Content-Type": "application/json"})
            mark("candidate_posted")
            sock = connection.sock
            if mode == "active_disconnect":
                log.wait_for(begin, lambda events: any(e["kind"] == "start" and e["tokens"] > 1400 for e in events),
                             "candidate slot start")
            timeout = 0 if mode == "active_disconnect" else QUEUE_OBSERVATION_SECONDS
            ready, _, _ = select.select([sock], [], [], timeout)
            if ready:
                raise ValueError("Candidate response became readable before the pre-first cancellation check")
            mark("candidate_socket_checked")

            candidate = None
            if disconnect:
                mark("candidate_disconnect_started")
                sock.shutdown(socket.SHUT_RDWR)
                connection.close()
                mark("candidate_disconnect_finished")
            else:
                candidate = executor.submit(finish_candidate, connection)
            follow_payload = payload_for("short", f"{offset + 2:03d}", output_length="short")
            follower = executor.submit(request, "follow", follow_payload, EXPECTED)
            if candidate is not None:
                candidate.result()
            follower.result()
            if holder is not None:
                holder.result()

    release_count = 3 if mode == "queued_complete" else 2
    events = log.after_releases(begin, release_count)
    starts = [e for e in events if e["kind"] == "start"]
    releases = {e["task"]: e for e in events if e["kind"] == "release"}
    cancellations = [e for e in events if e["kind"] == "cancel"]
    holder_start = next((e for e in starts if e["tokens"] == 889), None)
    candidate_starts = [e for e in starts if e["tokens"] > 1400]
    follow_start, = [e for e in starts if e["tokens"] == 190]
    if {e["task"] for e in starts} != releases.keys() or any(e["slot"] != 0 for e in starts):
        raise ValueError(f"Unexpected slot lifecycle in {mode}: {events}")
    if holder_start is not None and releases[holder_start["task"]]["observed_ms"] <= times["candidate_socket_checked"]:
        raise ValueError(f"Candidate was not checked while the slot holder was active: {events}")

    if mode == "active_disconnect":
        target, = candidate_starts
        release = releases[target["task"]]
        if ([e["task"] for e in cancellations] != [target["task"]]
                or not target["observed_ms"] <= times["candidate_disconnect_started"] <= release["observed_ms"]
                <= follow_start["observed_ms"]):
            raise ValueError(f"Candidate cancellation did not match its active task: {events}")
    elif mode == "queued_complete":
        target, = candidate_starts
        if cancellations or holder_start is None or target["observed_ms"] < releases[holder_start["task"]]["observed_ms"]:
            raise ValueError(f"Queued control did not start after the slot became free: {events}")
    elif holder_start is None or any(e["observed_ms"] < releases[holder_start["task"]]["observed_ms"]
                                     for e in candidate_starts):
        raise ValueError(f"Queued cancellation did not have a verified slot holder: {events}")

    result = {"repetition": repetition, "mode": mode, "client_ms": times, "server_log_events": events,
              "candidate_response_readable_at_check": False, "follow_reply_validated": True,
              "candidate_entered_slot": bool(candidate_starts), "candidate_reply_validated": not disconnect}
    if mode == "active_disconnect":
        result["disconnect_to_slot_release_log_ms"] = release["observed_ms"] - times["candidate_disconnect_started"]
    if holder_start is not None:
        result["holder_reply_validated"] = True
    print(f"Repetition {repetition}, {mode}: follow-up validated", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare queued and active disconnects before first text.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with dedicated_ollama(args.port, 1) as (log, origin, version, digest, configuration):
        trials = []
        for repetition in range(6):
            for mode in MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]:
                trials.append(run_trial(args.port, log, origin, repetition + 1, mode, 500 + len(trials) * 3))

    summary = {mode: {
        "trials": len([r for r in trials if r["mode"] == mode]),
        "candidate_entered_slot": sum(r["candidate_entered_slot"] for r in trials if r["mode"] == mode),
        "follow_replies_validated": sum(r["follow_reply_validated"] for r in trials if r["mode"] == mode),
    } for mode in MODES}
    summary["active_disconnect"]["median_disconnect_to_slot_release_log_ms"] = statistics.median(
        r["disconnect_to_slot_release_log_ms"] for r in trials if r["mode"] == "active_disconnect"
    )
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": f"http://127.0.0.1:{args.port}/api/chat",
        "ollama_version": version, "model": MODEL, "model_digest": digest, "server_configuration": configuration,
        "server_event_clock": "stderr line observation on the client monotonic clock",
        "queue_observation_ms": QUEUE_OBSERVATION_SECONDS * 1000, "trials": trials, "summary": summary,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
