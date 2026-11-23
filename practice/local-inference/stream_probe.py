import argparse
import json
import statistics
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


MODEL = "qwen3:8b"
EXPECTED = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
PROMPT = f"Copy the following line exactly, with no explanation or formatting:\n{EXPECTED}"
OPTIONS = {"temperature": 0, "num_predict": 64, "num_ctx": 2048}
COUNTERS = (
    "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_cached_count",
    "prompt_eval_duration", "eval_count", "eval_duration",
)


def read_stream(response, started, on_first_text=None):
    chunks = []
    first_text_ms = None
    events = 0
    for line in response:
        arrived_ms = (perf_counter() - started) * 1000
        if not line.strip():
            continue
        event = json.loads(line)
        if "error" in event:
            raise ValueError(f"Ollama stream error: {event['error']}")
        events += 1
        content = event.get("message", {}).get("content", "")
        if content:
            chunks.append(content)
            if first_text_ms is None and content.strip():
                if on_first_text is not None:
                    on_first_text()
                first_text_ms = arrived_ms
        if event.get("done") is True:
            if first_text_ms is None:
                raise ValueError("Stream ended without visible reply text")
            return event, "".join(chunks), first_text_ms, arrived_ms, events, len(chunks)
    raise ValueError("Stream ended before done:true")


def validate_reply(final, text, model, expected):
    if (final.get("model") != model or final.get("done") is not True
            or final.get("done_reason") != "stop" or text.strip() != expected):
        raise ValueError(f"Unexpected reply {text!r}, done reason {final.get('done_reason')!r}")


def probe(opener, url, payload, stream, expected=EXPECTED):
    request = urllib.request.Request(
        url, data=json.dumps({**payload, "stream": stream}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = perf_counter()
    with opener.open(request, timeout=180) as response:
        if response.status != 200:
            raise ValueError(f"Expected HTTP 200, got {response.status}")
        expected_type = "application/x-ndjson" if stream else "application/json"
        if response.headers.get_content_type() != expected_type:
            raise ValueError(f"Expected {expected_type}, got {response.headers.get_content_type()}")
        if stream:
            final, text, first_ms, done_ms, events, text_chunks = read_stream(response, started)
        else:
            body = response.read()
            done_ms = (perf_counter() - started) * 1000
            final = json.loads(body)
            text = final.get("message", {}).get("content", "")
            first_ms, events, text_chunks = None, None, None
    validate_reply(final, text, payload["model"], expected)
    return {
        "stream": stream, "http_status": 200, "reply": text, "done_reason": final["done_reason"],
        "first_text_ms": round(first_ms, 3) if first_ms is not None else None,
        "complete_ms": round(done_ms, 3), "events": events, "text_chunks": text_chunks,
        "server": {name: final.get(name) for name in COUNTERS},
    }


def main():
    parser = argparse.ArgumentParser(description="Compare serial Ollama streaming and JSON replies.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")

    base_url = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    payload = {
        "model": MODEL, "think": False, "options": OPTIONS,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    url = f"{base_url}/api/chat"
    print("Running warm-up request", flush=True)
    warmup = probe(opener, url, payload, stream=False)
    rows = []
    for pair in range(1, args.pairs + 1):
        for stream in ((True, False) if pair % 2 else (False, True)):
            rows.append({"pair": pair, **probe(opener, url, payload, stream)})
        print(f"Completed pair {pair}/{args.pairs}", flush=True)

    streamed = [row for row in rows if row["stream"]]
    buffered = [row for row in rows if not row["stream"]]
    summary = {
        "stream_first_text_median_ms": statistics.median(row["first_text_ms"] for row in streamed),
        "stream_complete_median_ms": statistics.median(row["complete_ms"] for row in streamed),
        "nonstream_complete_median_ms": statistics.median(row["complete_ms"] for row in buffered),
    }
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": url, "ollama_version": version, "request_common": payload, "expected_reply": EXPECTED,
        "warmup": warmup, "pairs": rows, "summary": summary,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
