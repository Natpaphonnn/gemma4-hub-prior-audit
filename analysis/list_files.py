"""List all competition files (paginated) into data/manifest.csv. Read-only; downloads nothing."""
import csv, io, subprocess, sys

SLUG = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-developer-agent"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/manifest.csv"

rows, token = [], None
while True:
    cmd = [".venv/bin/kaggle", "competitions", "files", SLUG, "-v", "--page-size", "200"]
    if token:
        cmd += ["--page-token", token]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    token = None
    if lines and lines[0].startswith("Next Page Token"):
        token = lines[0].split("=", 1)[1].strip()
        lines = lines[1:]
    rows += list(csv.DictReader(io.StringIO("\n".join(lines))))
    if not token:
        break

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} files -> {OUT}")
