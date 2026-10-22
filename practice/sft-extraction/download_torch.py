"""Download the official Windows cu128 wheel in ranges and verify index SHA-256."""
import concurrent.futures
import hashlib
import html
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

INDEX = "https://download.pytorch.org/whl/cu128/torch/"
FILENAME = "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
OUTPUT = Path("E:/Evren/Evren_work/tmp/sft-wheels") / FILENAME


def main():
    page = urllib.request.urlopen(INDEX, timeout=30).read().decode()
    links = [html.unescape(s) for s in re.findall(r'href="([^"]+)"', page)]
    link = next(s for s in links if urllib.parse.unquote(urllib.parse.urlparse(s).path).endswith("/" + FILENAME))
    url, fragment = urllib.parse.urldefrag(urllib.parse.urljoin(INDEX, link))
    expected = urllib.parse.parse_qs(fragment)["sha256"][0]
    assert urllib.parse.urlparse(url).hostname in {"download.pytorch.org", "download-r2.pytorch.org"}
    if OUTPUT.exists():
        with OUTPUT.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, "existing incomplete wheel; choose a new explicit path"
        print({"wheel": str(OUTPUT), "sha256": actual, "verified": True}, flush=True)
        return
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        total = int(response.headers["Content-Range"].split("/")[-1])
    chunk = 16 * 1024 * 1024

    def fetch(start):
        end = min(start + chunk, total) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"] == f"bytes {start}-{end}/{total}"
                    data = response.read()
                assert len(data) == end - start + 1
                return start, data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with OUTPUT.open("x+b") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, start) for start in range(0, total, chunk)}
        for future in concurrent.futures.as_completed(futures):
            start, data = future.result()
            stream.seek(start)
            stream.write(data)
            done += len(data)
            if done // (128 * 1024 * 1024) != (done - len(data)) // (128 * 1024 * 1024):
                print(f"Downloaded {done}/{total} bytes", flush=True)
            futures.remove(future)
        stream.flush()
    with OUTPUT.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == expected, "official SHA-256 mismatch; do not install"
    print({"wheel": str(OUTPUT), "sha256": actual, "verified": True}, flush=True)


if __name__ == "__main__":
    main()
