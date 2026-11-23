import argparse
import hashlib
import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from baseline_probe import PADDING
from concurrency_probe import LENGTHS, local_opener, run_round, summarize
from stream_probe import EXPECTED, MODEL, OPTIONS


LONG_REPLY = (
    "At sunrise, the station keeper opened the north gate and wrote the date in a blue notebook. "
    "A small ferry waited beside the landing while two crew members checked the ropes and counted the life jackets. "
    "Across the river, a gardener watered the rows of beans and placed a basket under the old pear tree. "
    "The baker brought six loaves to the counter and left a handwritten note beside the bread. "
    "By noon, the air was warmer and the wind had turned toward the hills. "
    "The last passenger folded a paper map, thanked the crew, and followed the stone path into town."
)
REPLIES = {"short": EXPECTED, "long": LONG_REPLY}
GENERATION_OPTIONS = {**OPTIONS, "num_predict": 256}
CONDITIONS = (("short", 1), ("short", 2), ("long", 2), ("long", 1))


def payload_for(length, label, *, output_length):
    background = f"Ignore this background passage:\n<background>\n{PADDING}\n</background>\n\n" if length == "long" else ""
    prompt = (
        f"{label}\n\n{background}Copy only the selected text exactly, including punctuation. "
        "Do not include the tag, quotes or any explanation.\n"
        f"<short>{EXPECTED}</short>\n<long>{LONG_REPLY}</long>\n"
        f"Selected text: {output_length}"
    )
    return {"model": MODEL, "think": False, "options": GENERATION_OPTIONS,
            "messages": [{"role": "user", "content": prompt}]}


def deduplicate_replies(rounds):
    replies = {}
    for run in rounds:
        for row in run["requests"]:
            if "reply" in row:
                reply = row.pop("reply")
                digest = hashlib.sha256(reply.encode("utf-8")).hexdigest()
                replies[digest] = reply
                row["reply_sha256"] = digest
    return replies


def main():
    parser = argparse.ArgumentParser(description="Compare output length and concurrency on one Ollama instance.")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT", default=11435)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    opener = local_opener()
    with opener.open(f"{base_url}/api/version", timeout=10) as response:
        version = json.load(response)["version"]
    with opener.open(f"{base_url}/api/tags", timeout=10) as response:
        model = next(item for item in json.load(response)["models"] if item["name"] == MODEL)
    url = f"{base_url}/api/chat"
    warmup = run_round(url, 0, 2, 900, build_payload=partial(payload_for, output_length="long"),
                       expected=LONG_REPLY)
    print(f"Warm-up: {warmup['successes']}/{len(LENGTHS)} passed", flush=True)

    rounds = []
    if not warmup["failures"]:
        for number in range(len(CONDITIONS)):
            for output_length, workers in CONDITIONS[number:] + CONDITIONS[:number]:
                offset = 200 + len(rounds) * len(LENGTHS)
                run = run_round(url, number + 1, workers, offset,
                                build_payload=partial(payload_for, output_length=output_length),
                                expected=REPLIES[output_length])
                rounds.append({"output_length": output_length, **run})
                print(f"Round {number + 1}, {output_length} output, concurrency {workers}: "
                      f"{run['successes']}/{len(LENGTHS)} passed", flush=True)

    summary = {
        output_length: {
            workers: summarize([run for run in rounds if run["output_length"] == output_length
                                and run["concurrency"] == workers])
            for workers in (1, 2)
        }
        for output_length in REPLIES
    } if rounds else {}
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "url": url,
        "ollama_version": version, "model": MODEL, "model_digest": model["digest"],
        "options": GENERATION_OPTIONS, "expected_replies": REPLIES,
        "actual_replies": deduplicate_replies([warmup, *rounds]),
        "prompt_templates": {length: payload_for(length, "{label}", output_length="{selected}")["messages"][0]["content"]
                             for length in ("short", "long")},
        "warmup": warmup, "rounds": rounds, "summary": summary,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if warmup["failures"]:
        raise SystemExit("Warm-up reply validation failed; diagnostics were saved")


if __name__ == "__main__":
    main()
