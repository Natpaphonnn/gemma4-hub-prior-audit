---
name: localize
description: BM25 localizer over /workspace source; returns top-k candidate files and functions with line ranges. Use before reading files when the statement does not name the file.
---

# Skill: localize

Find the files and functions most likely to contain the bug/feature described in the task, using a
fast BM25 search over the repository source. Use this **first**, before reading files or calling the
graph tools, whenever the problem statement does not already name the exact file to edit.

## When to use
- The statement describes a symptom but not a file path (the common case).
- You are unsure which file/function to open first.
- Do NOT use it to read file contents — it returns a ranked shortlist, then you `read_file` the top hits.

## How to run (works offline in the sandbox)
The script is pure standard-library Python. Write it to /tmp and run it so it never pollutes your patch:

    write_file("/tmp/localize.py", <contents of localize.py>)   # once
    run_command("python3 /tmp/localize.py --root /workspace --k 10 \"<the problem statement>\"")

Or, if the statement is long, put it in a file first:

    write_file("/tmp/ps.txt", "<the problem statement>")
    run_command("python3 /tmp/localize.py --root /workspace --k 10 --statement-file /tmp/ps.txt")

## What it returns (compact, safe for the 32k context)
- `top candidate files` — ranked list of file paths.
- `top candidate functions/classes` — `file:line_start-line_end  symbol  (score)`.

It excludes test files and configs from the candidates (you must not edit tests anyway).

## How to act on it
1. `read_file` the top 1–3 functions (use the printed line ranges to read only the relevant span).
2. If the top file looks right but the exact function is unclear, read the top 2–3 functions in that file.
3. Make the minimal edit, add/keep a targeted test if needed, then `submit_patch`.

## Notes
- It ranks by lexical overlap (BM25); if the statement names a symbol, that symbol usually ranks near
  the top. If results look off, add distinctive identifiers/error strings from the statement to the query.
- It does not run the code and costs one `run_command` call (~1–2 s on these repos).
