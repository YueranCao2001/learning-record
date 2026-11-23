import argparse
import http.client
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from concurrency_probe import LENGTHS, local_opener, run_round
from output_length_probe import GENERATION_OPTIONS, LONG_REPLY, deduplicate_replies
from queue_probe import LONG_OUTPUT_PAYLOAD
from stream_probe import COUNTERS, MODEL, read_stream, validate_reply


BUDGETS_MS = (40, 600, 4000)
ROUNDS = 4
AFTER_FIRST_IDLE_TIMEOUT_SECONDS = 30


def remaining(deadline):
    seconds = deadline - perf_counter()
    if seconds <= 0:
        raise TimeoutError("First-text deadline reached")
    return seconds


def timed_request(port, length, label, submitted, deadline, gate):
    result = {"length": length, "label": label, "http_start_ms": None, "http_status": None}
    acquired = False
    first_text_seen = False
    connection = None
    try:
        acquired = gate.acquire(timeout=remaining(deadline))
        if not acquired:
            raise TimeoutError("First-text deadline reached while waiting for admission")
        payload = LONG_OUTPUT_PAYLOAD(length, label)
        body = json.dumps({**payload, "stream": True}).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=remaining(deadline))
        started = perf_counter()
        result["http_start_ms"] = (started - submitted) * 1000
        connection.connect()
        sock = connection.sock
        sock.settimeout(remaining(deadline))
        connection.request("POST", "/api/chat", body=body, headers={"Content-Type": "application/json"})
        sock.settimeout(remaining(deadline))

        def accept_first_text():
            nonlocal first_text_seen
            remaining(deadline)
            first_text_seen = True
            sock.settimeout(AFTER_FIRST_IDLE_TIMEOUT_SECONDS)

        def response_lines(response):
            while True:
                if not first_text_seen:
                    sock.settimeout(remaining(deadline))
                line = response.readline()
                if not line:
                    return
                yield line

        with connection.getresponse() as response:
            result["http_status"] = response.status
            if response.status != 200 or response.headers.get_content_type() != "application/x-ndjson":
                raise ValueError(f"Unexpected HTTP response: {response.status} {response.headers.get_content_type()}")
            final, text, first_ms, _, _, _ = read_stream(response_lines(response), started, accept_first_text)
        validate_reply(final, text, MODEL, LONG_REPLY)
        result.update(outcome="completed", reply=text, first_text_ms=result["http_start_ms"] + first_ms,
                      done_reason=final["done_reason"], server={name: final.get(name) for name in COUNTERS})
    except TimeoutError as error:
        if result["http_start_ms"] is None:
            result["outcome"] = "deadline_before_http"
        else:
            result["outcome"] = "stream_idle_timeout" if first_text_seen else "deadline_after_http_start"
        result["detail"] = str(error)
    except (OSError, ValueError, http.client.HTTPException) as error:
        result.update(outcome="request_error", detail=f"{type(error).__name__}: {error}")
    finally:
        if connection is not None:
            connection.close()
        if acquired:
            gate.release()
    result["elapsed_ms"] = (perf_counter() - submitted) * 1000
    return result


def run_batch(port, number, budget_ms, offset):
    gate = BoundedSemaphore(2)
    with ThreadPoolExecutor(max_workers=len(LENGTHS)) as executor:
        submitted = perf_counter()
        deadline = submitted + budget_ms / 1000
        tasks = [executor.submit(timed_request, port, length, f"{offset + idx:03d}", submitted, deadline, gate)
                 for idx, length in enumerate(LENGTHS)]
        rows = [task.result() for task in tasks]
    counts = Counter(row["outcome"] for row in rows)
    print(f"Round {number}, {budget_ms} ms: {dict(counts)}", flush=True)
    return {"round": number, "budget_ms": budget_ms, "requests": rows}


def main():
    parser = argparse.ArgumentParser(description="Measure an absolute first-text budget from batch arrival.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    opener = local_opener()
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    with opener.open(f"{base_url}/api/tags", timeout=10) as response:
        model = next(item for item in json.load(response)["models"] if item["name"] == MODEL)
    warmup = run_round(f"{base_url}/api/chat", 0, 2, 900, build_payload=LONG_OUTPUT_PAYLOAD, expected=LONG_REPLY)
    print(f"Warm-up: {warmup['successes']}/{len(LENGTHS)} passed", flush=True)

    batches = []
    if not warmup["failures"]:
        for number in range(1, ROUNDS + 1):
            for budget_ms in BUDGETS_MS:
                offset = 600 + len(batches) * len(LENGTHS)
                batches.append(run_batch(args.port, number, budget_ms, offset))

    summary = {budget: dict(Counter(row["outcome"] for batch in batches if batch["budget_ms"] == budget
                                     for row in batch["requests"])) for budget in BUDGETS_MS}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": f"{base_url}/api/chat",
        "ollama_version": version, "model": MODEL, "model_digest": model["digest"],
        "options": GENERATION_OPTIONS, "expected_reply": LONG_REPLY,
        "after_first_idle_timeout_seconds": AFTER_FIRST_IDLE_TIMEOUT_SECONDS,
        "actual_replies": deduplicate_replies([warmup, *batches]),
        "prompt_templates": {length: LONG_OUTPUT_PAYLOAD(length, "{label}")["messages"][0]["content"]
                             for length in ("short", "long")},
        "warmup": warmup, "batches": batches, "summary": summary,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if warmup["failures"]:
        raise SystemExit("Warm-up reply validation failed; diagnostics were saved")


if __name__ == "__main__":
    main()
