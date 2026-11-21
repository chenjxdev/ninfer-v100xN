"""Discriminating benchmark aimed at what DaisyRose was trained for.

The earlier battery saturated on code and reasoning, and separated the models
only on mechanical word counting. This battery targets the published claims:
long-horizon agent work (Terminal Bench 4.1: 12.0 vs 5.6), harder coding, and
long-horizon state tracking.

Three item families, all objectively scored:
  term   - a real task in a sandboxed directory; the model's shell command is
           executed and the resulting filesystem state is checked.
  code   - harder algorithms, executed against real test cases.
  state  - long operation sequences whose final state must be reported exactly.

Sampling: greedy (temperature 0) for determinism. --thinking toggles the
reasoning mode, which is the mode the model card evaluates in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

URL = "http://172.16.3.88:8080/v1/chat/completions"
MODEL = "qwen3.8-27b"
SANDBOX_ROOT = Path("/tmp/ninfer-agent-sandbox")


def ask(prompt, max_tokens=1200, thinking=False, temperature=0.0):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": thinking,
    }).encode()
    request = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    wall = time.perf_counter() - started
    text = payload["choices"][0]["message"].get("content") or ""
    usage = payload.get("usage") or {}
    return text, wall, usage.get("completion_tokens") or 0


# --------------------------------------------------------------- shell utils

DANGEROUS = re.compile(
    r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/|rm\s+-[a-z]*f[a-z]*r[a-z]*\s+/|"
    r"\bmkfs\b|\bdd\s+if=|\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b|"
    r"\bsudo\b|\bsu\s|\bchmod\s+777\s+/|\bchown\b|>\s*/dev/|"
    r":\(\)\s*\{|\bfork\b|\bkillall\b|\bpkill\b|\biptables\b|\bmount\b",
    re.I)


def shell_ok(command):
    if DANGEROUS.search(command):
        return False
    if re.search(r"\b(curl|wget|ssh|scp|nc|ncat|telnet|ftp|pip|apt|apt-get)\b", command):
        return False
    return True


def extract_commands(text):
    """Take the model's shell answer: prefer a fenced block, else the raw text."""
    blocks = re.findall(r"```(?:bash|sh|shell)?[ \t]*\n?(.*?)```", text, re.S)
    body = blocks[0] if blocks else text
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    return "\n".join(lines).strip()


def run_in_sandbox(command, workdir):
    if not shell_ok(command):
        return False, "rejected by sandbox policy"
    script = (
        "set -e\n"
        "ulimit -t 15 -f 20480 -n 128\n"
        "cd %s\n"
        "export PATH=/usr/local/bin:/usr/bin:/bin\n"
        "%s\n" % (sh_quote(str(workdir)), command)
    )
    try:
        proc = subprocess.run(
            ["timeout", "-k", "5", "25", "bash", "-c", script],
            capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode != 0:
        return False, "exit %d: %s" % (proc.returncode, (proc.stderr or proc.stdout)[-300:])
    return True, proc.stdout[-300:]


def sh_quote(value):
    return "'" + value.replace("'", "'\\''") + "'"


# ------------------------------------------------------------------- tasks

def setup_logs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(1, 121):
        status = 500 if i % 7 == 0 else (404 if i % 5 == 0 else 200)
        lines.append('10.0.0.%d - - "GET /x%d HTTP/1.1" %d %d' % (i % 250, i, status, i * 13))
    (directory / "access.log").write_text("\n".join(lines) + "\n")


def setup_files(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "a.txt").write_text("alpha\n")
    (directory / "b.log").write_text("b\n" * 40)
    (directory / "c.log").write_text("c\n" * 90)
    (directory / "big.dat").write_text("z" * 50000)
    (directory / "notes.md").write_text("note\n")


def setup_config(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(
        {"host": "0.0.0.0", "port": 8080, "debug": False, "workers": 4}, indent=2) + "\n")
    (directory / "keep.txt").write_text("do not delete\n")


def setup_tree(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("src/main.py", "print('main')\n"),
                          ("src/util.py", "def util():\n    return 42\n"),
                          ("src/lib/deep.py", "def deep():\n    return 7\n"),
                          ("docs/readme.md", "# docs\n"),
                          ("data/values.csv", "a,b\n1,2\n3,4\n")):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


TERM_TASKS = [
    ("term-count500", setup_logs,
     "The current directory contains access.log. Write a shell command that counts the lines "
     "whose HTTP status is 500 and writes that number into count.txt. Output only the command.",
     lambda d: (d / "count.txt").exists() and (d / "count.txt").read_text().strip() == "17"),
    ("term-movelogs", setup_files,
     "The current directory contains a.txt, b.log, c.log, big.dat and notes.md. Write a shell "
     "command that creates a subdirectory logs/ and moves every .log file into it. Output only "
     "the command.",
     lambda d: (d / "logs" / "b.log").exists() and (d / "logs" / "c.log").exists()
     and not (d / "b.log").exists() and not (d / "c.log").exists()
     and (d / "a.txt").exists() and (d / "big.dat").exists()),
    ("term-largest", setup_files,
     "The current directory contains several files. Write a shell command that writes the name "
     "of the largest regular file in the current directory (not recursive) into biggest.txt. "
     "Output only the command.",
     lambda d: (d / "biggest.txt").exists()
     and (d / "biggest.txt").read_text().strip().split("/")[-1] == "big.dat"),
    ("term-port", setup_config,
     "The current directory contains config.json. Write a shell command that changes only the "
     "port value to 8081 and leaves the rest of the JSON valid and unchanged. Output only the "
     "command.",
     lambda d: _json_port_ok(d)),
    ("term-pyfiles", setup_tree,
     "The current directory contains a source tree. Write a shell command that counts how many "
     ".py files exist anywhere below the current directory and writes that number into "
     "pycount.txt. Output only the command.",
     lambda d: (d / "pycount.txt").exists() and (d / "pycount.txt").read_text().strip() == "3"),
    ("term-grep", setup_tree,
     "The current directory contains a source tree. Write a shell command that writes the "
     "relative paths of all files containing the string 'return' into found.txt, one per line. "
     "Output only the command.",
     lambda d: _found_ok(d)),
    ("term-sumcsv", setup_tree,
     "The current directory contains data/values.csv with two numeric columns and a header row. "
     "Write a shell command that sums the second column (skipping the header) and writes the "
     "total into total.txt. Output only the command.",
     lambda d: (d / "total.txt").exists() and (d / "total.txt").read_text().strip() == "6"),
    ("term-backup", setup_files,
     "The current directory contains a.txt. Write a shell command that creates a copy named "
     "a.txt.bak and then replaces every occurrence of 'alpha' in a.txt with 'beta', leaving "
     "a.txt.bak unchanged. Output only the command.",
     lambda d: (d / "a.txt.bak").exists() and (d / "a.txt.bak").read_text() == "alpha\n"
     and "beta" in (d / "a.txt").read_text() and "alpha" not in (d / "a.txt").read_text()),
]


def _json_port_ok(directory):
    try:
        data = json.loads((directory / "config.json").read_text())
    except Exception:
        return False
    return data == {"host": "0.0.0.0", "port": 8081, "debug": False, "workers": 4}


def _found_ok(directory):
    path = directory / "found.txt"
    if not path.exists():
        return False
    lines = {ln.strip().lstrip("./") for ln in path.read_text().splitlines() if ln.strip()}
    return {"src/util.py", "src/lib/deep.py"}.issubset(lines) and "src/main.py" not in lines


# --------------------------------------------------------------- code items

def code_blocks(text):
    return re.findall(r"```(?:python|py)?[ \t]*\n?(.*?)```", text, re.S)


def run_code(source, name, tests):
    namespace = {}
    try:
        exec(compile(source, "<model>", "exec"), namespace)
    except Exception as error:
        return False, "exec: %s" % error
    fn = namespace.get(name)
    if fn is None:
        return False, "missing %s" % name
    for entry in tests:
        expected, args = entry[0], entry[1:]
        try:
            got = fn(*args)
        except Exception as error:
            return False, "%s raised %r" % (name, error)
        if got != expected:
            return False, "%s(*%r) -> %r, want %r" % (name, args, got, expected)
    return True, "ok"


CODE_ITEMS = [
    ("code-edit-distance", "edit_distance",
     "Write a Python function `edit_distance(a, b)` returning the Levenshtein distance between "
     "two strings. Output only one Python code block.",
     [(0, "", ""), (3, "kitten", "sitting"), (2, "flaw", "lawn"), (0, "abc", "abc")]),
    ("code-lis", "lis",
     "Write a Python function `lis(nums)` returning the length of the longest strictly "
     "increasing subsequence. Output only one Python code block.",
     [(4, [10, 9, 2, 5, 3, 7, 101, 18]), (1, [7, 7, 7, 7]), (0, []), (6, [1, 3, 6, 7, 9, 4, 10, 5, 6])]),
    ("code-coins", "min_coins",
     "Write a Python function `min_coins(coins, amount)` returning the fewest coins needed to "
     "make amount, or -1 if impossible. Output only one Python code block.",
     [(3, [1, 2, 5], 11), (-1, [2], 3), (0, [1], 0), (4, [2, 5, 10, 1], 27)]),
    ("code-nqueens", "count_nqueens",
     "Write a Python function `count_nqueens(n)` returning the number of distinct solutions to "
     "the n-queens puzzle. Output only one Python code block.",
     [(2, 4), (10, 5), (4, 6), (1, 1)]),
    ("code-minwindow", "min_window",
     "Write a Python function `min_window(s, t)` returning the shortest substring of s that "
     "contains every character of t (with multiplicity), or '' if none exists. Output only one "
     "Python code block.",
     [("BANC", "ADOBECODEBANC", "ABC"), ("", "a", "aa"), ("a", "a", "a")]),
    ("code-topo", "topo_sort",
     "Write a Python function `topo_sort(n, edges)` returning a valid topological ordering of "
     "nodes 0..n-1 for the directed edges (u, v), or [] if the graph has a cycle. Output only "
     "one Python code block.",
     "topo"),
    ("code-lru", "LRU",
     "Write a Python class `LRU` with `__init__(self, capacity)`, `get(self, key)` returning "
     "the value or -1, and `put(self, key, value)` evicting the least recently used item when "
     "full. Output only one Python code block.",
     "lru"),
    ("code-bigsum", "add_big",
     "Write a Python function `add_big(a, b)` that adds two non-negative integers given as "
     "decimal strings and returns the sum as a string, without converting them with int(). "
     "Output only one Python code block.",
     [("579", "123", "456"), ("1000", "999", "1"), ("0", "0", "0")]),
]

TOPO_EDGES = [(4, [(0, 1), (1, 2), (2, 3)]), (3, [(0, 1), (1, 0)])]


def check_code(item_id, text):
    for cid, name, prompt, tests in CODE_ITEMS:
        if cid != item_id:
            continue
        blocks = code_blocks(text)
        if not blocks:
            return False, "no code block"
        if tests == "topo":
            return check_topo(blocks[0])
        if tests == "lru":
            return check_lru(blocks[0])
        return run_code(blocks[0], name, tests)
    return False, "unknown item"


def check_topo(source):
    namespace = {}
    try:
        exec(compile(source, "<m>", "exec"), namespace)
        fn = namespace["topo_sort"]
        order = fn(4, [(0, 1), (1, 2), (2, 3)])
        if sorted(order) != [0, 1, 2, 3]:
            return False, "not a permutation: %r" % (order,)
        pos = {node: i for i, node in enumerate(order)}
        if not all(pos[u] < pos[v] for u, v in [(0, 1), (1, 2), (2, 3)]):
            return False, "violates an edge: %r" % (order,)
        if fn(3, [(0, 1), (1, 0)]) != []:
            return False, "cycle not detected"
        return True, "ok"
    except Exception as error:
        return False, repr(error)


def check_lru(source):
    namespace = {}
    try:
        exec(compile(source, "<m>", "exec"), namespace)
        cls = namespace["LRU"]
        cache = cls(2)
        cache.put(1, 1)
        cache.put(2, 2)
        if cache.get(1) != 1:
            return False, "get(1) after put"
        cache.put(3, 3)
        if cache.get(2) != -1:
            return False, "2 should have been evicted"
        cache.put(4, 4)
        if cache.get(1) != -1 or cache.get(3) != 3 or cache.get(4) != 4:
            return False, "eviction order wrong"
        return True, "ok"
    except Exception as error:
        return False, repr(error)


# -------------------------------------------------------------- state items

def stack_program(steps):
    ops = []
    values = []
    for i in range(steps):
        if i % 4 == 2 and values:
            ops.append("pop")
            values.pop()
        else:
            value = (i * 7 + 3) % 20
            ops.append("push %d" % value)
            values.append(value)
    return ops, values


STATE_ITEMS = []
for steps in (8, 16, 24, 32):
    ops, remaining = stack_program(steps)
    STATE_ITEMS.append((
        "state-stack-%d" % steps,
        "Simulate a stack that starts empty. Apply these operations in order: %s. "
        "At the end, output the stack contents from bottom to top as a comma separated list of "
        "numbers, or 'empty' if it is empty. Output only that list." % ", ".join(ops),
        remaining,
    ))


def check_state(expected, text):
    numbers = [int(x) for x in re.findall(r"-?\d+", text)]
    if not expected:
        return not numbers or "empty" in text.lower()
    return numbers == expected


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--families", default="term,code,state")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    families = {f.strip() for f in args.families.split(",") if f.strip()}

    results = []
    if SANDBOX_ROOT.exists():
        shutil.rmtree(SANDBOX_ROOT)
    SANDBOX_ROOT.mkdir(parents=True)

    def record(item_id, family, prompt, ok, wall, tokens, text, detail=""):
        results.append({"id": item_id, "family": family, "ok": bool(ok),
                        "wall": round(wall, 2), "out_tokens": tokens,
                        "answer": text.strip()[:400], "detail": detail})
        print("[%2d] %-20s %-5s %-4s %6.2fs %5d tok %s" % (
            len(results), item_id, family, "PASS" if ok else "FAIL", wall, tokens,
            (detail or text.strip().replace("\n", " "))[:80]))

    if "term" in families:
        for task_id, setup, prompt, check in TERM_TASKS:
            workdir = SANDBOX_ROOT / task_id
            setup(workdir)
            text, wall, tokens = ask(prompt, 8000 if args.thinking else 1200, args.thinking)
            command = extract_commands(text)
            ok_run, detail = run_in_sandbox(command, workdir)
            ok = ok_run and check(workdir)
            record(task_id, "term", prompt, ok, wall, tokens, text,
                   "" if ok else (detail if not ok_run else "state mismatch: " + command[:120]))

    if "code" in families:
        for item_id, name, prompt, tests in CODE_ITEMS:
            text, wall, tokens = ask(prompt, 2500, args.thinking)
            ok, detail = check_code(item_id, text)
            record(item_id, "code", prompt, ok, wall, tokens, text, detail)

    if "state" in families:
        for item_id, prompt, expected in STATE_ITEMS:
            text, wall, tokens = ask(prompt, 8000 if args.thinking else 1200, args.thinking)
            ok = check_state(expected, text)
            record(item_id, "state", prompt, ok, wall, tokens, text,
                   "" if ok else "expected %s" % (expected,))

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    print()
    print("=== %s (thinking=%s) ===" % (args.tag, args.thinking))
    by_family = {}
    for r in results:
        e = by_family.setdefault(r["family"], [0, 0])
        e[0] += 1
        e[1] += int(r["ok"])
    for family in sorted(by_family):
        count, good = by_family[family]
        print("  %-8s %d/%d" % (family, good, count))
    print("  %-8s %d/%d" % ("TOTAL", passed, total))
    print("  wall %.1fs  out %d tok" % (sum(r["wall"] for r in results),
                                        sum(r["out_tokens"] for r in results)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"tag": args.tag, "thinking": args.thinking,
                       "passed": passed, "total": total, "results": results},
                      handle, ensure_ascii=False, indent=2)
        print("  saved %s" % args.out)


if __name__ == "__main__":
    main()
