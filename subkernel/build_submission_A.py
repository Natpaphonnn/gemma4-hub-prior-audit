"""Build submission.zip for the Gemma 4 Developer Agent competition (config A = baseline).

Config A is the public walkthrough agent 'Version 2' by Kaggle user zhukovoleksiy
(https://www.kaggle.com/code/zhukovoleksiy/gemma-4-walkthrough-first-submission, LB 0.10), reused with
attribution per the competition's public-sharing rules, plus a safe explicit 4-min/task eval_config.
This notebook writes /kaggle/working/submission.zip and validates it. Submit THIS notebook.
"""
import json, os, re, zipfile
from pathlib import Path
import yaml

FILES = {
 "agent.yaml": "name: swe_coder\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Fixes one repository issue with a small, verified source patch.\ninstruction: !include prompts/system.md\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\n  - get_code_neighbors\n  - search_similar_code\n  - get_code_subgraph\n  - agent_tool:\n      config_path: sub_agents/code_analyzer.yaml\n      skip_summarization: true\ngenerate_content_config: !include configs/sampling.yaml\n",
 "eval_config.yaml": "# Per-task cap (safe for the shared 12h wall-clock incl. setup). The agent self-terminates via\n# submit_patch well before this; a 129-task run finishes in well under 12h (walkthrough: ~80 min).\nevaluation:\n  timeout_seconds: 120\n  max_time_minutes: 4\n  max_turns: 80\n",
 "configs/sampling.yaml": "temperature: 0.2\ntop_p: 0.95\ntop_k: 40\nmax_output_tokens: 8192\nseed: 42\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: false\n",
 "prompts/system.md": "You are an autonomous software engineer fixing one issue in the Python repository at /workspace. No one will answer questions. Keep working with your tools until the fix is in place, then call `submit_patch()`.\n\n## How you are graded\n- Hidden tests run on your changes in a fresh checkout. Only source changes in the working tree count, and a careful best-effort fix beats no patch.\n- If time runs out, the working tree is graded as it is, so never revert a plausible fix.\n- Test files used by the hidden tests are reset before grading, so editing tests never helps. Never edit tests, `pytest.ini` or `conftest.py`.\n\n## Environment\n- Offline, with dependencies installed. Do not install packages.\n- /workspace is a git repository at the base commit.\n- Available: git, grep, find, sed, awk, python3. Not available: rg (ripgrep), tree.\n- Commands time out after 300 seconds and output is cut at 5,000 characters. `read_file` returns at most 150 lines, so read focused ranges.\n- `get_status()` and `submit_patch()` are free.\n\n## Workflow\n1. Understand. Work out the expected and the actual behaviour. Some issues are pull-request descriptions: skip checklists and template text, keep the substance. Note exact names, messages and values.\n2. Localize. Call `code_analyzer` with the issue text; it replies with LOCATION, ROOT CAUSE and FIX PLAN. Confirm by reading those lines. If it is unsure or wrong, search yourself: `git grep -n \"<identifier>\" -- '*.py' | head -30`.\n3. Reproduce. Write a minimal script to /tmp/repro.py that shows the problem, and run it.\n4. Fix. Edit with `edit_file`: copy `old_string` exactly from the file, including indentation, and keep it short but unique. Fix the root cause, handle the edge cases the issue names, and keep public signatures unchanged. Documentation examples under `docs_src/` count as source when the issue concerns them.\n5. Verify. After each edit run `python3 -m py_compile <file>` and rerun /tmp/repro.py, then run the targeted tests described below.\n6. Submit. Check `git status --short` and `git diff`, remove anything unintended, then call `submit_patch()` as your last tool call and reply with one line on what changed.\n\n## Tests\n- Run only the test file or test that exercises your change: `python3 -m pytest tests/test_x.py -q -x -k <name>`, or `python3 -m unittest tests.test_x`. Find test files with `git ls-files | grep test_<name>`.\n- Never run bare `pytest`, `pytest .` or `python3 -m unittest discover`. A whole suite can take minutes and use up your time.\n- Existing tests may already fail for reasons unrelated to the issue, such as missing fixtures, no network access or import errors. Ignore those failures: do not fix them, stub them or change test code. Judge your fix by the tests that exercise it.\n\n## Code-graph tools\n- `get_code_neighbors` and `get_code_subgraph` show synchronous call relationships only: the graph has only \"calls\" edges and no async functions.\n- `search_similar_code` takes a symbol name such as `APIKeyHeader`, not a sentence, and its ranking is imprecise. Confirm every hit by reading the code.\n- If a graph tool returns an error, stop using it. For async code and everything else, use `git grep` and `read_file`.\n\n## Pace\n- Call `get_status()` every 8 or so tool calls. When a quarter of the time remains, stop exploring and go straight to fix, verify and submit.\n- Keep outputs short: `head`, `grep -n`, `pytest -q`.\n- If an edit fails twice, reread the exact lines and retry with a smaller snippet. Keep each edit small so the tool call is not cut off.\n",
 "prompts/CREDIT.txt": "Config A (baseline) is the public walkthrough agent 'Version 2' by Kaggle user zhukovoleksiy,\n'Gemma 4 | Walkthrough & First Submission' (scored 0.10 on public LB), reused with attribution per\ncompetition public-sharing rules. Source: https://www.kaggle.com/code/zhukovoleksiy/gemma-4-walkthrough-first-submission\n",
 "prompts/analyzer.md": "You are `code_analyzer`, a read-only code navigator. You never modify files. Given an issue, find exactly where it must be fixed.\n\n## Tools\n- `run_command` for read-only commands only: `git grep -n`, `grep -rn`, `sed -n 'START,ENDp' FILE`, `ls`, `git log -p -S \"<text>\"`. ripgrep (rg) is not installed.\n- `read_file` with tight line ranges to confirm what you found.\n- `get_code_neighbors` and `get_code_subgraph` for synchronous call relationships. The graph has only \"calls\" edges and no async functions.\n- `search_similar_code` takes a symbol name such as `APIKeyHeader` or `routing.get_request_handler`, not a sentence, and its ranking is imprecise. Confirm every hit by reading the code.\n- If a graph tool returns an error, stop using it and search the source instead.\n\n## Method\n1. Pull the identifiers out of the issue: function and class names, error messages, file paths, options. Pull-request descriptions may include template text; ignore it.\n2. Search for each one, then follow the code until you reach the line where the behaviour diverges from what the issue expects.\n3. Confirm by reading the code. Never guess line numbers.\n\n## Answer (at most 250 words, nothing else)\nLOCATION: <path>:<start>-<end> (<function or class>)\nROOT CAUSE: <one or two sentences>\nFIX PLAN: <the concrete change>\nRELATED: <other places needing the same change, or \"none\">\nTESTS: <existing test files that exercise this code>\nCONFIDENCE: high | medium | low\n",
 "sub_agents/code_analyzer.yaml": "name: code_analyzer\nmodel: gemma-4-31b-it-qat-w4a16-ct\ndescription: Read-only code navigator. Given an issue, returns where to fix it, the root cause and a fix plan.\ninstruction: !include ../prompts/analyzer.md\ntools:\n  - run_command\n  - read_file\n  - get_code_neighbors\n  - get_code_subgraph\n  - search_similar_code\ngenerate_content_config: !include ../configs/sampling.yaml\n"
}

WORKING = Path("/kaggle/working")
BUNDLE = WORKING / "bundle"
import shutil; shutil.rmtree(BUNDLE, ignore_errors=True)
for rel, text in FILES.items():
    p = BUNDLE / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text)

# validate (mirrors the harness submission checks)
ROOT_CONFIGS = ["agent.yaml","agent.yml","root_agent.yaml","root_agent.yml"]
ALLOWED = {".yaml",".yml",".md",".txt",".py",".json",".safetensors"}
yaml.SafeLoader.add_constructor("!include", lambda l,n: l.construct_scalar(n))
files=[p for p in BUNDLE.rglob("*") if p.is_file()]
roots=[n for n in ROOT_CONFIGS if (BUNDLE/n).is_file()]
assert len(roots)==1, roots
assert not any(p.is_symlink() for p in BUNDLE.rglob("*"))
bad=[p.name for p in files if p.suffix not in ALLOWED]; assert not bad, bad
assert sum(p.stat().st_size for p in files) < 3*2**30
models=set()
for c in (p for p in files if p.suffix in (".yaml",".yml")):
    t=c.read_text(); d=yaml.safe_load(t) or {}
    for inc in re.findall(r"!include\s+(\S+)", t):
        assert (c.parent/inc).is_file(), f"{c.name}: missing include {inc}"
    if "adapter" in d:
        assert (BUNDLE/"adapters"/d["adapter"]/"adapter_model.safetensors").is_file()
    if "model" in d: models.add(d["model"])
assert models == {"gemma-4-31b-it-qat-w4a16-ct"}, models
print("VALIDATION OK:", roots[0], len(files), "files, model", models)

ZIP = WORKING / "submission.zip"
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
    for p in files: z.write(p, p.relative_to(BUNDLE))
print("wrote", ZIP, ZIP.stat().st_size, "bytes")
with zipfile.ZipFile(ZIP) as z: print("contents:", z.namelist())
