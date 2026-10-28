"""Ground-truth localisation labels: which graph nodes does a task's patch actually modify?

We do NOT trust hunk-header `def` names. Instead we tie the label to the graph representation
the retrievers see: a node is GOLD iff a distinctive pre-patch line of the patch (a removed line,
or a context line for pure-addition hunks) appears verbatim in that node's stored `text`.

This makes "function recall" mean exactly: did the retriever surface a node the agent must edit?

Leakage note: the patch is the LABEL. It is used only here to build gold sets, never as a
retriever feature.
"""
import re
from pathlib import Path

MODULE_STRIP = re.compile(r"^(src|docs_src)/")


def file_to_module(path: str) -> str:
    p = MODULE_STRIP.sub("", path)
    p = p[:-3] if p.endswith(".py") else p
    return p.replace("/", ".").removesuffix(".__init__")


def iter_file_hunks(patch: str):
    """Yield (path, is_new, [old_lines], [added_lines]) per file in a unified diff."""
    path, is_new, old, add, have = None, False, [], [], False
    for line in patch.splitlines():
        if line.startswith("+++ "):
            if have:
                yield path, is_new, old, add
            p = line[4:].strip()
            path = p[2:] if p.startswith("b/") else p
            is_new, old, add, have = False, [], [], True
        elif line.startswith("--- "):
            if "/dev/null" in line:
                is_new = True
        elif have and line.startswith("@@"):
            continue
        elif have and line.startswith("-") and not line.startswith("---"):
            old.append(line[1:])
        elif have and line.startswith("+") and not line.startswith("+++"):
            add.append(line[1:])
        elif have and line.startswith(" "):
            old.append(line[1:])
    if have:
        yield path, is_new, old, add


def distinctive(lines, min_len=8):
    """Pre-patch lines worth matching (skip blanks, closers, decorators-only)."""
    out = []
    for ln in lines:
        s = ln.strip()
        if len(s) >= min_len and s not in (")", "):", "]", "}") and not s.startswith("#"):
            out.append(s)
    return out


DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")


def gold_nodes(patch: str, nodes):
    """Return (gold_fn, gold_mod, meta, gold_hdr). gold_hdr is an INDEPENDENT label (def/class names
    on changed/context lines) used only to cross-check gold_fn. nodes: list of {name,text}."""
    by_name = {n["name"]: n for n in nodes}
    names = list(by_name)
    gold_fn, gold_mod, gold_hdr = set(), set(), set()
    n_files = n_new = n_mod_in_graph = 0
    for path, is_new, old, add in iter_file_hunks(patch):
        if not path.endswith(".py"):
            n_files += 1
            continue
        n_files += 1
        if is_new:
            n_new += 1
            continue
        mod = file_to_module(path)
        in_mod = [nm for nm in names if nm == mod or nm.startswith(mod + ".")]
        if not in_mod:
            continue
        n_mod_in_graph += 1
        gold_mod.add(mod)
        probe = distinctive(old) or distinctive(add)  # removed lines preferred
        matched = {nm for nm in in_mod if any(s in (by_name[nm]["text"] or "") for s in probe)}
        gold_fn |= matched
        # if nothing matched but module is present, fall back to module-level nodes
        if not matched:
            gold_fn |= {nm for nm in in_mod if nm.count(".") == mod.count(".") + 1}
        hdr_syms = {m.group(1) for ln in old + add for m in [DEF_RE.match(ln)] if m}
        gold_hdr |= {nm for nm in in_mod if nm.split(".")[-1] in hdr_syms}
    meta = dict(n_files=n_files, n_new_files=n_new, n_mod_in_graph=n_mod_in_graph,
                all_new=(n_new == n_files and n_files > 0))
    return gold_fn, gold_mod, meta, gold_hdr


def module_of_node(name: str, modules: set) -> str:
    """Longest known module prefix of a node name (which file does this node live in)."""
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in modules:
            return cand
    return ".".join(parts[:-1]) if len(parts) > 1 else name
