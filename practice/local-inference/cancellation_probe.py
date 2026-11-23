import argparse
import http.client
import json
import os
import re
import socket
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Condition, Event, Thread
from time import perf_counter, sleep
from urllib.error import URLError

from concurrency_probe import local_opener
from output_length_probe import LONG_REPLY, payload_for
from queue_probe import LONG_OUTPUT_PAYLOAD
from stream_probe import EXPECTED, MODEL, probe, read_stream, validate_reply


EVENT_PATTERNS = {
    "configuration": re.compile(r"initializing, n_slots = (\d+), n_ctx_slot = (\d+)"),
    "start": re.compile(r"id\s+(\d+)\s+\| task (\d+) \| new prompt, .*task\.n_tokens = (\d+)"),
    "cancel": re.compile(r"cancel task, id_task = (\d+)"),
    "release": re.compile(r"id\s+(\d+)\s+\| task (\d+) \| stop processing: n_tokens = (\d+)"),
}


class SlotLog:
    def __init__(self, stream, origin):
        self.events = []
        self.condition = Condition()
        self.reader = Thread(target=self.read, args=(stream, origin), daemon=True)
        self.reader.start()

    def read(self, stream, origin):
        for line in stream:
            now = (perf_counter() - origin) * 1000
            for kind, pattern in EVENT_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    fields = tuple(map(int, match.groups()))
                    event = {"kind": kind, "observed_ms": now, "line": line.strip()}
                    if kind == "configuration":
                        event.update(slots=fields[0], context_per_slot=fields[1])
                    elif kind == "cancel":
                        event["task"] = fields[0]
                    else:
                        event.update(slot=fields[0], task=fields[1], tokens=fields[2])
                    with self.condition:
                        self.events.append(event)
                        self.condition.notify_all()
                    break

    def after_releases(self, index, count):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: sum(e["kind"] == "release" for e in self.events[index:]) >= count,
                timeout=10,
            )
            if not ready:
                raise RuntimeError(f"Missing slot release evidence: {self.events[index:]}")
            return self.events[index:].copy()


class ClosedAfterFirstText(Exception):
    pass


def open_chat(port, payload, posted=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        body = json.dumps({**payload, "stream": True}).encode("utf-8")
        connection.request("POST", "/api/chat", body, {"Content-Type": "application/json"})
        if posted is not None:
            posted()
        response = connection.getresponse()
        if response.status != 200 or response.headers.get_content_type() != "application/x-ndjson":
            response.close()
            raise ValueError(f"Unexpected HTTP response: {response.status}")
        return connection, response
    except BaseException:
        connection.close()
        raise


def match_trial_events(events, dual_slot, cancel):
    starts = [e for e in events if e["kind"] == "start"]
    releases = {e["task"]: e for e in events if e["kind"] == "release"}
    cancellations = [e for e in events if e["kind"] == "cancel"]
    expected_starts = [190, 889] if dual_slot else [889]
    if (sorted(e["tokens"] for e in starts[:-1]) != expected_starts or starts[-1]["tokens"] != 190
            or len(starts) != len(releases) or {e["task"] for e in starts} != releases.keys()
            or any(e["slot"] != releases[e["task"]]["slot"] for e in starts)):
        raise ValueError(f"Trial prompts and releases were not uniquely matched: {events}")
    target = next(e for e in starts[:-1] if e["tokens"] == (190 if dual_slot else 889))
    follow = starts[-1]
    previous = next((e for e in starts[:-1] if e["slot"] == follow["slot"]), None)
    if (previous is None or releases[previous["task"]]["observed_ms"] > follow["observed_ms"]
            or [e["task"] for e in cancellations] != ([target["task"]] if cancel else [])):
        raise ValueError(f"Follow-up and cancellation events did not match: {events}")

    sibling = next((e for e in starts[:-1] if e["task"] != target["task"]), None)
    if sibling is not None and sibling["slot"] == target["slot"]:
        raise ValueError(f"The two initial streams did not use distinct slots: {events}")
    return target, follow, sibling, releases, cancellations


def run_trial(port, log, origin, pair, cancel, offset, dual_slot=False):
    begin = len(log.events)
    times = {}
    follow_posted = Event()
    sibling_posted = Event()
    sibling_first_text = Event()

    def mark(name):
        times[name] = (perf_counter() - origin) * 1000

    def consume(role, payload, expected, posted, first_text=None):
        def on_posted():
            mark(f"{role}_posted")
            posted.set()

        def on_first_text():
            mark(f"{role}_first_text")
            if first_text is not None:
                first_text.set()

        connection, response = open_chat(port, payload, on_posted)
        with closing(connection), response:
            final, text, *_ = read_stream(response, perf_counter(), on_first_text)
            validate_reply(final, text, MODEL, expected)
        mark(f"{role}_completed")

    def wait_for_signal(signal, future, message):
        if not signal.wait(timeout=3):
            if future.done():
                future.result()
            raise TimeoutError(message)

    with ThreadPoolExecutor(max_workers=2 if dual_slot else 1) as executor:
        sibling_future = None
        if dual_slot:
            sibling_future = executor.submit(consume, "sibling", LONG_OUTPUT_PAYLOAD("long", f"{offset + 2:03d}"),
                                            LONG_REPLY, sibling_posted, sibling_first_text)
            wait_for_signal(sibling_posted, sibling_future, "The second stream did not post")
        target, response = open_chat(port, LONG_OUTPUT_PAYLOAD("short" if dual_slot else "long", f"{offset:03d}"))
        follower = None
        with closing(target), response:
            def on_first_text():
                nonlocal follower
                mark("target_first_text")
                if sibling_future is not None:
                    wait_for_signal(sibling_first_text, sibling_future, "The second stream did not return first text")
                follow_payload = payload_for("short", f"{offset + 1:03d}", output_length="short")
                follower = executor.submit(consume, "follow", follow_payload, EXPECTED, follow_posted)
                wait_for_signal(follow_posted, follower, "Follow-up request did not post while the target was active")
                if cancel:
                    mark("target_disconnect_started")
                    target.sock.shutdown(socket.SHUT_RDWR)
                    response.close()
                    target.close()
                    mark("target_disconnect_finished")
                    raise ClosedAfterFirstText

            try:
                final, text, *_ = read_stream(response, perf_counter(), on_first_text)
                validate_reply(final, text, MODEL, LONG_REPLY)
                mark("target_completed")
            except ClosedAfterFirstText:
                pass
        follower.result()
        if sibling_future is not None:
            sibling_future.result()

    events = log.after_releases(begin, 3 if dual_slot else 2)
    target_start, follow_start, sibling, releases, cancellations = match_trial_events(events, dual_slot, cancel)
    target_release = releases[target_start["task"]]

    measurements = {"follow_posted_to_slot_ms": follow_start["observed_ms"] - times["follow_posted"],
                    "follow_posted_to_first_text_ms": times["follow_first_text"] - times["follow_posted"]}
    if cancel:
        measurements["disconnect_to_cancel_log_ms"] = cancellations[0]["observed_ms"] - times["target_disconnect_started"]
        measurements["disconnect_to_release_log_ms"] = target_release["observed_ms"] - times["target_disconnect_started"]
    if sibling is not None:
        measurements["sibling_first_text_to_complete_ms"] = times["sibling_completed"] - times["sibling_first_text"]
        if cancel and (follow_start["slot"] != target_start["slot"]
                       or releases[sibling["task"]]["observed_ms"] <= follow_start["observed_ms"]):
            raise ValueError(f"Follow-up did not reuse the cancelled slot while the sibling was active: {events}")
    print(f"Pair {pair}, {'disconnect' if cancel else 'complete'}: follow-up validated", flush=True)
    return {"pair": pair, "target_cancelled": cancel, "client_ms": times, "server_log_events": events,
            "task_roles": {"target": target_start["task"], "follow": follow_start["task"],
                           **({"sibling": sibling["task"]} if sibling is not None else {})},
            "measurements": measurements, "follow_reply_validated": True,
            **({"sibling_reply_validated": True} if sibling is not None else {})}


def wait_ready(server, opener, base_url):
    limit = perf_counter() + 20
    while perf_counter() < limit and server.poll() is None:
        try:
            with opener.open(f"{base_url}/api/version", timeout=1) as response:
                return json.load(response)["version"]
        except (URLError, TimeoutError):
            sleep(.15)
    raise RuntimeError("Dedicated Ollama process did not become ready")


def main():
    parser = argparse.ArgumentParser(description="Observe Ollama slot release after a streamed HTTP disconnect.")
    parser.add_argument("--port", type=int, choices=range(1024, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dual-slot", action="store_true", help="Keep a second stream active during the test.")
    args = parser.parse_args()
    slots = 2 if args.dual_slot else 1

    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    base_url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{args.port}", "OLLAMA_NUM_PARALLEL": str(slots)}
    origin = perf_counter()
    with subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8", errors="replace",
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as server:
        log = SlotLog(server.stderr, origin)
        try:
            opener = local_opener()
            version = wait_ready(server, opener, base_url)
            with opener.open(f"{base_url}/api/tags", timeout=10) as response:
                model = next(item for item in json.load(response)["models"] if item["name"] == MODEL)
            probe(opener, f"{base_url}/api/chat", LONG_OUTPUT_PAYLOAD("long", "900"), stream=True, expected=LONG_REPLY)
            warmup_events = log.after_releases(0, 1)
            configuration, = [event for event in warmup_events if event["kind"] == "configuration"]
            if configuration["slots"] != slots or configuration["context_per_slot"] != 2048:
                raise ValueError(f"Expected {slots} slots with 2048 tokens each: {configuration}")
            rows = []
            for pair in range(1, 7):
                for cancel in ((True, False) if pair % 2 else (False, True)):
                    rows.append(run_trial(args.port, log, origin, pair, cancel, 600 + len(rows) * 3, args.dual_slot))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            log.reader.join(timeout=5)

    summary = {}
    for label, cancel in (("disconnect", True), ("complete", False)):
        selected = [row for row in rows if row["target_cancelled"] == cancel]
        summary[label] = {name: statistics.median(row["measurements"][name] for row in selected)
                          for name in selected[0]["measurements"]}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": f"{base_url}/api/chat",
        "ollama_version": version, "model": MODEL, "model_digest": model["digest"],
        "server_configuration": configuration,
        "server_event_clock": "stderr line observation on the client monotonic clock",
        "trials": rows, "summary_median_ms": summary,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
