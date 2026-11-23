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


def run_trial(port, log, origin, pair, cancel, offset):
    begin = len(log.events)
    times = {}
    posted = Event()

    def mark(name):
        times[name] = (perf_counter() - origin) * 1000

    def follow():
        def on_posted():
            mark("follow_posted")
            posted.set()

        connection, response = open_chat(port, payload_for("short", f"{offset + 1:03d}", output_length="short"), on_posted)
        with closing(connection), response:
            final, text, *_ = read_stream(response, perf_counter(), lambda: mark("follow_first_text"))
            validate_reply(final, text, MODEL, EXPECTED)
        mark("follow_completed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        target, response = open_chat(port, LONG_OUTPUT_PAYLOAD("long", f"{offset:03d}"))
        follower = None
        with closing(target), response:
            def on_first_text():
                nonlocal follower
                mark("target_first_text")
                follower = executor.submit(follow)
                if not posted.wait(timeout=3):
                    if follower.done():
                        follower.result()
                    raise TimeoutError("Follow-up request did not post while the target was active")
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

    events = log.after_releases(begin, 2)
    starts = [e for e in events if e["kind"] == "start"]
    releases = [e for e in events if e["kind"] == "release"]
    if [e["tokens"] for e in starts] != [889, 190] or len(releases) != 2:
        raise ValueError(f"Target and follow-up prompts were not uniquely matched: {events}")
    target_start, follow_start = starts
    target_release, follow_release = releases
    cancellations = [e for e in events if e["kind"] == "cancel"]
    if ([e["task"] for e in starts] != [e["task"] for e in releases]
            or any(e["slot"] != target_start["slot"] for e in starts + releases)
            or [e["task"] for e in cancellations] != ([target_start["task"]] if cancel else [])
            or not target_start["observed_ms"] <= target_release["observed_ms"] <= follow_start["observed_ms"]
            or follow_release["task"] != follow_start["task"]):
        raise ValueError(f"Target and follow-up slot events did not match: {events}")

    measurements = {"follow_posted_to_slot_ms": follow_start["observed_ms"] - times["follow_posted"],
                    "follow_posted_to_first_text_ms": times["follow_first_text"] - times["follow_posted"]}
    if cancel:
        measurements["disconnect_to_cancel_log_ms"] = cancellations[0]["observed_ms"] - times["target_disconnect_started"]
        measurements["disconnect_to_release_log_ms"] = target_release["observed_ms"] - times["target_disconnect_started"]
    print(f"Pair {pair}, {'disconnect' if cancel else 'complete'}: follow-up validated", flush=True)
    return {"pair": pair, "target_cancelled": cancel, "client_ms": times, "server_log_events": events,
            "measurements": measurements, "follow_reply_validated": True}


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
    args = parser.parse_args()

    with socket.socket() as port_check:
        port_check.bind(("127.0.0.1", args.port))
    base_url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{args.port}", "OLLAMA_NUM_PARALLEL": "1"}
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
            if configuration["slots"] != 1 or configuration["context_per_slot"] != 2048:
                raise ValueError(f"Expected one 2048-token slot: {configuration}")
            rows = []
            for pair in range(1, 7):
                for cancel in ((True, False) if pair % 2 else (False, True)):
                    rows.append(run_trial(args.port, log, origin, pair, cancel, 600 + len(rows) * 2))
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
