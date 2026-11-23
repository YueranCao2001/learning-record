import argparse
import hashlib
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
from concurrency_probe import local_opener
from output_length_probe import LONG_REPLY, payload_for
from queue_probe import LONG_OUTPUT_PAYLOAD
from stream_probe import EXPECTED, MODEL, probe, read_stream, validate_reply


MODES = ("queued_disconnect", "active_disconnect", "queued_complete")
QUEUE_OBSERVATION_SECONDS = 0.08
SWEEP_PADDING_LINES = (2, 8, 16, 32)


def candidate_payload(label, padding_lines):
    payload = LONG_OUTPUT_PAYLOAD("short", label)
    message = payload["messages"][0]
    heading, task = message["content"].split("\n\n", 1)
    background = "\n".join((PADDING.splitlines() * 2)[:padding_lines])
    message["content"] = f"{heading}\n\nIgnore this background:\n{background}\n\n{task}"
    return payload


def run_trial(port, log, origin, repetition, mode, offset, padding_lines=32, follow_label=None):
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
            payload = candidate_payload(f"{offset + 1:03d}", padding_lines)
            connection.request("POST", "/api/chat", json.dumps({**payload, "stream": True}).encode("utf-8"),
                               {"Content-Type": "application/json"})
            mark("candidate_posted")
            sock = connection.sock
            if mode == "active_disconnect":
                log.wait_for(begin, lambda events: any(e["kind"] == "start" and e["tokens"] > 190 for e in events),
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
            follow_payload = payload_for("short", follow_label or f"{offset + 2:03d}", output_length="short")
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
    holder_start = next((e for e in starts if e["tokens"] == 889), None) if holder is not None else None
    candidate_starts = [e for e in starts if e["tokens"] > 190 and e is not holder_start]
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
        result["candidate_prompt_sha256"] = hashlib.sha256(payload["messages"][0]["content"].encode("utf-8")).hexdigest()
        result["disconnect_to_slot_release_log_ms"] = release["observed_ms"] - times["candidate_disconnect_started"]
        result["slot_start_log_to_disconnect_ms"] = times["candidate_disconnect_started"] - target["observed_ms"]
        result["disconnect_to_cancel_log_ms"] = cancellations[0]["observed_ms"] - times["candidate_disconnect_started"]
        result["candidate_prompt_tokens"] = target["tokens"]
        result["padding_lines"] = padding_lines
        result["initial_cached_tokens"] = next(
            e["tokens"] for e in events if e["kind"] == "prompt_position" and e["task"] == target["task"]
        )
    if holder_start is not None:
        result["holder_reply_validated"] = True
    print(f"Repetition {repetition}, {mode}: follow-up validated", flush=True)
    return result


def summarize_sweep(trials):
    summary = {}
    for lines in SWEEP_PADDING_LINES:
        rows = [row for row in trials if row["padding_lines"] == lines]
        prompt_tokens, = {row["candidate_prompt_tokens"] for row in rows}
        release_ms = [row["disconnect_to_slot_release_log_ms"] for row in rows]
        summary[lines] = {
            "trials": len(rows), "prompt_tokens": prompt_tokens,
            "initial_cached_tokens": sorted({row["initial_cached_tokens"] for row in rows}),
            "median_slot_start_log_to_disconnect_ms": statistics.median(row["slot_start_log_to_disconnect_ms"] for row in rows),
            "median_disconnect_to_cancel_log_ms": statistics.median(row["disconnect_to_cancel_log_ms"] for row in rows),
            "median_disconnect_to_slot_release_log_ms": statistics.median(release_ms),
            "release_log_range_ms": [min(release_ms), max(release_ms)],
            "follow_replies_validated": sum(row["follow_reply_validated"] for row in rows),
        }
    return summary


def run_cache_pairs(port, log, origin):
    opener = local_opener()
    url = f"http://127.0.0.1:{port}/api/chat"
    trials = []
    for pair in range(1, 9):
        offset = 800 + (pair - 1) * 4
        cold = run_trial(port, log, origin, pair, "active_disconnect", offset)
        before_primer = len(log.events)
        payload = candidate_payload(f"{offset + 1:03d}", 32)
        primer = probe(opener, url, payload, stream=True, expected=LONG_REPLY)
        primer_events = log.after_releases(before_primer, 1)
        warm = run_trial(port, log, origin, pair, "active_disconnect", offset, follow_label=f"{offset + 3:03d}")
        if (cold["candidate_prompt_tokens"] != 1570 or warm["candidate_prompt_tokens"] != 1570
                or cold["candidate_prompt_sha256"] != warm["candidate_prompt_sha256"]
                or cold["initial_cached_tokens"] > 16 or warm["initial_cached_tokens"] < 0.9 * 1570
                or primer["server"]["prompt_eval_count"] != 1570):
            raise ValueError(f"Pair {pair} did not match the expected prompt lengths or cache conditions")
        trials.extend([
            {"cache_condition": "new_prefix", **cold},
            {"cache_condition": "primed_repeat", "primer": {
                "prompt_tokens": primer["server"]["prompt_eval_count"],
                "generated_tokens": primer["server"]["eval_count"],
                "reply_validated": True, "server_log_events": primer_events,
            }, **warm},
        ])
    return trials


def summarize_cache_pairs(trials):
    summary = {}
    for condition in ("new_prefix", "primed_repeat"):
        rows = [row for row in trials if row["cache_condition"] == condition]
        releases = [row["disconnect_to_slot_release_log_ms"] for row in rows]
        summary[condition] = {
            "trials": len(rows), "prompt_tokens": 1570,
            "initial_cached_tokens": sorted({row["initial_cached_tokens"] for row in rows}),
            "median_slot_start_log_to_disconnect_ms": statistics.median(row["slot_start_log_to_disconnect_ms"] for row in rows),
            "median_disconnect_to_cancel_log_ms": statistics.median(row["disconnect_to_cancel_log_ms"] for row in rows),
            "median_disconnect_to_slot_release_log_ms": statistics.median(releases),
            "release_log_range_ms": [min(releases), max(releases)],
            "follow_replies_validated": sum(row["follow_reply_validated"] for row in rows),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare queued and active disconnects before first text.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument("--length-sweep", action="store_true",
                             help="Disconnect an active candidate before first text at four prompt lengths.")
    experiments.add_argument("--cache-pair", action="store_true",
                             help="Compare new prefixes and primed repeats of the same 1570-token prompts.")
    args = parser.parse_args()

    with dedicated_ollama(args.port, 1) as (log, origin, version, digest, configuration):
        trials = []
        if args.cache_pair:
            trials = run_cache_pairs(args.port, log, origin)
        elif args.length_sweep:
            for repetition in range(8):
                rotation = repetition % len(SWEEP_PADDING_LINES)
                for lines in SWEEP_PADDING_LINES[rotation:] + SWEEP_PADDING_LINES[:rotation]:
                    trials.append(run_trial(args.port, log, origin, repetition + 1, "active_disconnect",
                                            700 + len(trials) * 3, padding_lines=lines))
        else:
            for repetition in range(6):
                for mode in MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]:
                    trials.append(run_trial(args.port, log, origin, repetition + 1, mode, 500 + len(trials) * 3))

    if args.cache_pair:
        summary = summarize_cache_pairs(trials)
    elif args.length_sweep:
        summary = summarize_sweep(trials)
    else:
        summary = {mode: {
            "trials": sum(r["mode"] == mode for r in trials),
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
        "trials": trials, "summary": summary,
    }
    if args.cache_pair:
        report["cache_pair_order"] = ["new_prefix", "completed_primer", "primed_repeat"]
    elif args.length_sweep:
        report["padding_lines"] = SWEEP_PADDING_LINES
    else:
        report["queue_observation_ms"] = QUEUE_OBSERVATION_SECONDS * 1000
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
