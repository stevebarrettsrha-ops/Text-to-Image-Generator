"""The process primitives, on their own.

These are the parts of the engine kit with no HTTP around them: who holds a
port, what a process is called, what the system said when we asked it to stop.
They are cheap to check and the whole takeover rests on them.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bootstrap  # noqa: E402

from harness import Suite, comfy, free_port  # noqa: E402


def orphan(code: str, ready: Path | None = None) -> int:
    """Start a process that belongs to nobody — spawned by a child that exits
    at once, so it is reparented and reaped like the stray ComfyUI it stands
    in for. Returns its pid, once `ready` (if given) says it is set up."""
    # The grandchild must not inherit the pipe this reads, or the wait below
    # blocks until it exits — which is the opposite of the point.
    launcher = ("import subprocess, sys\n"
                f"p = subprocess.Popen([sys.executable, '-c', {code!r}],\n"
                "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
                "    stdin=subprocess.DEVNULL, start_new_session=True)\n"
                "print(p.pid)\n")
    out = subprocess.run([sys.executable, "-c", launcher],
                         capture_output=True, text=True, timeout=30)
    pid = int(out.stdout.strip())
    for _ in range(100):        # a signal sent before it is set up proves nothing
        if alive(pid) and (ready is None or ready.exists()):
            return pid
        time.sleep(0.05)
    return pid


def alive(pid: int, within: float = 0.0) -> bool:
    deadline = time.time() + within
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        if time.time() >= deadline:
            return True
        time.sleep(0.1)


def run(slow: bool = False) -> Suite:
    s = Suite("units")

    # -- who holds the port -------------------------------------------------
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    port = sock.getsockname()[1]
    try:
        pids = bootstrap.port_pids(port)
        s.check("port_pids finds the process listening on a port",
                os.getpid() in pids, f"saw {pids}")
    finally:
        sock.close()
    s.equal("and reports nothing for a port no one holds",
            bootstrap.port_pids(free_port()), [])

    # -- what it is called --------------------------------------------------
    mine = bootstrap.pid_cmdline(os.getpid())
    s.check("pid_cmdline reads a real command line",
            "python" in mine.lower() or "run.py" in mine, mine[:80])
    s.equal("and says nothing about a pid that does not exist",
            bootstrap.pid_cmdline(4_000_000), "")

    # -- stopping things ----------------------------------------------------
    # Orphans on a port belong to nobody, so the tests use grandchildren too:
    # a process of our own would linger as a zombie after SIGTERM and never
    # look "stopped", which is an artefact of being ours and nothing else.
    victim = orphan("import time; time.sleep(120)")
    s.equal("kill_pid reports what the system said",
            bootstrap.kill_pid(victim), "stopped")
    s.check("and the process really is gone", not alive(victim))
    flag = Path(tempfile.mkdtemp(prefix="ig-kill-")) / "ready"
    stubborn = orphan("import pathlib, signal, time\n"
                      "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                      f"pathlib.Path({str(flag)!r}).write_text('up')\n"
                      "time.sleep(120)", ready=flag)
    s.equal("one that ignores the polite ask is killed outright",
            bootstrap.kill_pid(stubborn), "sent SIGKILL")
    s.check("and that one is gone too", not alive(stubborn, within=5))
    s.equal("stopping something already finished says so",
            bootstrap.kill_pid(victim), "already gone")

    # -- what is answering --------------------------------------------------
    s.equal("comfy_stats says None when nothing answers",
            bootstrap.comfy_stats(f"http://127.0.0.1:{free_port()}"), None)
    with comfy(MOCK_COMFY_ROOT="/opt/some/ComfyUI") as mock:
        stats = bootstrap.comfy_stats(mock.url) or {}
        s.check("and hands back the system block, argv and all",
                stats.get("argv", [""])[0] == "/opt/some/ComfyUI/main.py",
                str(stats)[:80])

    # -- the console ring buffer --------------------------------------------
    proc = bootstrap.ComfyProcess()
    proc.note("taking it over")
    s.equal("note() lands in the same console as the engine's own output",
            proc.tail(1), ["[Ideogram Studio] taking it over"])
    for i in range(2100):
        proc.note(str(i))
    s.check("the console is a ring buffer, not a leak",
            len(proc.lines) <= 2000 and proc.tail(1) == ["[Ideogram Studio] 2099"],
            f"{len(proc.lines)} lines")
    s.check("a process nobody started is not alive", proc.alive() is False)

    # -- waiting ------------------------------------------------------------
    seen: list[float] = []
    began = time.time()
    got = bootstrap.wait_for_comfy(f"http://127.0.0.1:{free_port()}", timeout=5,
                                   on_wait=lambda elapsed, limit: seen.append(elapsed))
    s.check("wait_for_comfy gives up and narrates the wait",
            got is False and len(seen) >= 2 and time.time() - began >= 4,
            f"{len(seen)} callbacks")
    return s
