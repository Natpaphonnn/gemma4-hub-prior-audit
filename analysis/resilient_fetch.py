"""Patient, resumable, single-worker fetch for graphs+embeddings.

Sequential (no parallelism → no rate-limit storm), per-file timeout, inter-file delay,
indefinite retry with capped backoff. Skips files already on disk. Fetches GRAPHS FIRST
(we already have most embeddings), so the probe becomes runnable as early as possible.

Usage: .venv/bin/python analysis/resilient_fetch.py [delay_seconds] [per_file_timeout]
"""
import csv
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLUG = "gemma-4-developer-agent"
DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
TIMEOUT = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0

rows = list(csv.DictReader(open(ROOT / "data/manifest.csv")))
# graphs first, then embeddings
wanted = [r["name"] for r in rows if r["name"].startswith("graphs/")] + \
         [r["name"] for r in rows if r["name"].startswith("embeddings/")]
todo = [f for f in wanted if not (ROOT / "data" / f).exists()]
print(f"{len(todo)} files to fetch (graphs first). delay={DELAY}s timeout={TIMEOUT}s", flush=True)

done = 0
for f in todo:
    d = ROOT / "data" / os.path.dirname(f)
    d.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while not (ROOT / "data" / f).exists():
        attempt += 1
        try:
            subprocess.run([os.environ.get("KAGGLE_BIN", "kaggle"),
                            "competitions", "download", SLUG, "-f", f, "-p", str(d), "-q"],
                           timeout=TIMEOUT, capture_output=True, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            pass
        z = d / (os.path.basename(f) + ".zip")
        if z.exists():
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
                z.unlink()
            except zipfile.BadZipFile:
                z.unlink(missing_ok=True)
        if (ROOT / "data" / f).exists():
            break
        back = min(60, 5 * attempt)
        if attempt % 5 == 0:
            print(f"  {f}: {attempt} attempts, backing off {back}s (rate-limited?)", flush=True)
        time.sleep(back)
    done += 1
    if done % 10 == 0:
        ng = len(list((ROOT / "data/graphs").glob("*.json")))
        ne = len(list((ROOT / "data/embeddings").glob("*.npz")))
        print(f"progress: {done}/{len(todo)} done | graphs={ng}/127 emb={ne}/127", flush=True)
    time.sleep(DELAY)

ng = len(list((ROOT / "data/graphs").glob("*.json")))
ne = len(list((ROOT / "data/embeddings").glob("*.npz")))
print(f"FETCH COMPLETE graphs={ng}/127 emb={ne}/127", flush=True)
