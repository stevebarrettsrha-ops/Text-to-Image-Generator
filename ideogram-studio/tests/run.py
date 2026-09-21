#!/usr/bin/env python3
"""Run the suite.

    python tests/run.py              everything available
    python tests/run.py gate units   just those
    python tests/run.py --list       what there is

Needs nothing beyond what Ideogram Studio itself needs (plus node for the
inline-script check, which steps aside when missing).
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Ordered cheapest first, so a plain mistake shows up before the slow ones.
MODULES = [
    ("gate", "the checks CLAUDE.md asks for after any edit"),
    ("units", "the process primitives the takeover rests on"),
    ("api", "the engine kit end to end, against real engine processes"),
]


def gate() -> tuple[int, list[str]]:
    """py_compile every module, and node --check the inline script.

    A missing function in that script kills every control on the page without
    a word, so this is not optional.
    """
    passed, failures = 0, []
    sources = ["server.py", "comfy.py", "bootstrap.py", "manager.py"]
    done = subprocess.run([sys.executable, "-m", "py_compile", *sources],
                          cwd=ROOT, capture_output=True, text=True)
    if done.returncode == 0:
        passed += 1
        print(f"  ok   {' '.join(sources)} all compile")
    else:
        failures.append("python compile")
        print(f"  FAIL python compile — {done.stderr.strip()[:200]}")

    page = (ROOT / "web" / "index.html").read_text()
    script = "\n".join(re.findall(r"<script>(.*?)</script>", page, re.S))
    scratch = HERE / ".inline.js"
    scratch.write_text(script)
    try:
        done = subprocess.run(["node", "--check", str(scratch)],
                              capture_output=True, text=True)
        if done.returncode == 0:
            passed += 1
            print("  ok   the inline script parses")
        else:
            failures.append("inline script")
            print(f"  FAIL inline script — {done.stderr.strip()[:200]}")
    except FileNotFoundError:
        print("  --   node is not installed, so the inline script was not checked")
    finally:
        scratch.unlink(missing_ok=True)

    # every $("id") must name something that exists, or the page dies at boot
    markup = page[:page.index("<script>")] + page[page.rindex("</script>"):]
    defined = set(re.findall(r'\bid="([^"]+)"', markup))
    missing = sorted({u for u in re.findall(r'\$\("([^"]+)"\)', script)
                      if u not in defined})
    if missing:
        failures.append("missing ids")
        print(f"  FAIL the script reaches for ids that do not exist: {missing}")
    else:
        passed += 1
        print("  ok   every id the script reaches for exists")

    # a control with no listener is a button that does nothing: the panel was
    # added and the wiring forgotten, which no amount of reading catches
    def wired(control: str) -> bool:
        # $("id")…addEventListener close by, directly or via a wiring loop
        for m in re.finditer(re.escape(f'$("{control}")'), script):
            if "addEventListener" in script[m.end():m.end() + 260]:
                return True
        for m in re.finditer(r'\[(?:\[?"[^"]+"[^\]]*\]?,?\s*)+\]\.forEach',
                             script):
            if f'"{control}"' in m.group(0) and \
                    "addEventListener" in script[m.end():m.end() + 400]:
                return True
        return False

    interactive = set(re.findall(r'<button[^>]*\bid="([^"]+)"', markup))
    interactive |= set(re.findall(r'<input type="range" id="([^"]+)"', markup))
    # read when they are used rather than listened to: the brush slider is
    # sampled by every stamp, so it has nothing to react to
    passive = {"edit-brush"}
    deaf = sorted(c for c in interactive - passive if not wired(c))
    if deaf:
        failures.append("deaf controls")
        print(f"  FAIL controls no listener ever touches: {deaf}")
    else:
        passed += 1
        print("  ok   every interactive control has a listener")

    if (ROOT / "run.sh").exists():
        done = subprocess.run(["bash", "-n", str(ROOT / "run.sh")],
                              capture_output=True, text=True)
        if done.returncode == 0:
            passed += 1
            print("  ok   run.sh parses")
        else:
            failures.append("run.sh")
            print(f"  FAIL run.sh — {done.stderr.strip()[:200]}")
    return passed, failures


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    slow = "--slow" in sys.argv
    if "--list" in sys.argv:
        for name, what in MODULES:
            print(f"  {name:10s} {what}")
        return 0

    chosen = [m for m in MODULES if not args or m[0] in args]
    if args and not chosen:
        print(f"No such test: {' '.join(args)}. Try --list.")
        return 2

    results, started = [], time.time()
    for name, what in chosen:
        print(f"\n=== {name} — {what} ===")
        began = time.time()
        if name == "gate":
            passed, failures = gate()
        else:
            module = importlib.import_module(f"test_{name}")
            skip = getattr(module, "available", lambda: "")()
            if skip:
                print(f"  --   skipped: {skip}")
                results.append((name, 0, [], 0.0, skip))
                continue
            try:
                suite = module.run(slow)
                passed, failures = suite.passed, suite.failures
            except Exception as exc:  # noqa: BLE001
                import traceback

                import harness
                partial = harness.CURRENT
                passed = partial.passed if partial else 0
                failures = list(partial.failures) if partial else []
                failures.append(f"{type(exc).__name__}: {exc}")
                print(f"  FAIL {name} stopped early — {type(exc).__name__}: "
                      f"{str(exc)[:120]}")
                print("       " + traceback.format_exc().strip()
                      .replace("\n", "\n       ")[-600:])
        results.append((name, passed, failures, time.time() - began, ""))

    print("\n" + "=" * 62)
    total = sum(r[1] for r in results)
    broken = [r for r in results if r[2]]
    for name, passed, failures, took, skipped in results:
        if skipped:
            print(f"  {name:10s} skipped — {skipped}")
        else:
            state = "ok" if not failures else f"{len(failures)} FAILED"
            print(f"  {name:10s} {passed:3d} passed  {state:>12s}  {took:5.1f}s")
    print("=" * 62)
    print(f"  {total} checks passed in {time.time()-started:.0f}s"
          + (f", {sum(len(r[2]) for r in broken)} failed" if broken else ""))
    for name, _, failures, _, _ in broken:
        for failure in failures:
            print(f"    {name}: {failure}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
