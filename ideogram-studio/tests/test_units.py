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
from unittest.mock import patch

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
        if not bootstrap.pid_alive(pid):
            return False
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

    # /proc is Linux-specific.  A Unix platform without it must use the
    # portable signal probe rather than declaring every process dead.
    with patch.object(bootstrap.platform, "system", return_value="Darwin"):
        s.check("pid_alive falls back correctly when procfs is unavailable",
                bootstrap.pid_alive(os.getpid()))
    s.check("pid_alive rejects process-group sentinel ids",
            not bootstrap.pid_alive(0) and not bootstrap.pid_alive(-1))

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

    # -- a moved app folder: stale saved paths are found again ---------------
    real_app = bootstrap.APP_DIR
    with tempfile.TemporaryDirectory() as tmp:
        app = Path(tmp) / "text-to-image-generator-main" / real_app.name
        (app / "ComfyUI" / "models").mkdir(parents=True)
        (app / "ComfyUI" / "main.py").write_text("")
        bootstrap.APP_DIR = app
        try:
            old = ("C:\\AI\\text-to-image-generator\\" + real_app.name
                   + "\\ComfyUI")
            moved = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=old,
                         models_dir=old + "\\models",
                         python="C:\\gone\\python.exe")
            notes = bootstrap.heal_paths(moved)
            s.equal("a stale ComfyUI path is rebased onto the moved app",
                    moved["comfy_dir"], str(app / "ComfyUI"))
            s.equal("a stale models path follows it",
                    moved["models_dir"], str(app / "ComfyUI" / "models"))
            s.equal("a vanished Python path is cleared, not kept",
                    moved["python"], "")
            s.check("each repair is reported", len(notes) == 3, str(notes))
            blank = dict(bootstrap.DEFAULT_CONFIG)
            bootstrap.heal_paths(blank)
            s.equal("an empty config adopts the ComfyUI inside the app",
                    blank["comfy_dir"], str(app / "ComfyUI"))
            s.check("a valid config is left alone",
                    bootstrap.heal_paths(blank) == [])
        finally:
            bootstrap.APP_DIR = real_app

    # -- set up on Windows, opened on Linux (a drive shared by both) -------
    # The Windows venv and portable interpreter are still on disk, so they
    # "exist"; every run of them is [Errno 8] Exec format error and the
    # engine never starts.
    import platform as _platform
    import manager
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(_platform, "system", return_value="Linux"):
        app = Path(tmp) / "text-to-image-generator-main" / "ideogram-studio"
        comfy_dir = app / "ComfyUI"
        (comfy_dir / "models").mkdir(parents=True)
        (comfy_dir / "main.py").write_text("")
        win_py = app / "comfy-venv" / "Scripts" / "python.exe"
        win_py.parent.mkdir(parents=True)
        win_py.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)
        elf = app / "elf"
        elf.write_bytes(b"\x7fELF\x02\x01")
        s.check("a Windows .exe is foreign on Linux",
                bootstrap.foreign_binary(win_py))
        s.check("an ELF binary is native on Linux",
                not bootstrap.foreign_binary(elf))
        s.check("a script or unreadable file is never called foreign",
                not bootstrap.foreign_binary(comfy_dir / "main.py")
                and not bootstrap.foreign_binary(app / "nope"))
        with patch.object(_platform, "system", return_value="Windows"):
            s.check("an ELF binary is foreign on Windows",
                    bootstrap.foreign_binary(elf))

        cfg_w = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(comfy_dir),
                     models_dir=str(comfy_dir / "models"), python=str(win_py),
                     comfy_url="http://127.0.0.1:9", setup_complete=True)
        s.equal("comfy_python does not hand out the foreign interpreter",
                bootstrap.comfy_python(cfg_w), "")
        notes = bootstrap.heal_paths(cfg_w)
        s.equal("a saved interpreter built for Windows is cleared on Linux",
                cfg_w["python"], "")
        s.check("and the reason is reported",
                any("another operating system" in n for n in notes),
                "; ".join(notes))
        moved = dict(cfg_w, python="D:\\AI\\text-to-image-generator-main\\"
                     "ideogram-studio\\comfy-venv\\Scripts\\python.exe")
        with patch.object(bootstrap, "APP_DIR", app):
            bootstrap.heal_paths(moved)
        s.equal("a rebased path that lands on the Windows python is cleared",
                moved["python"], "")
        s.equal("the Windows venv is left alone; Linux gets its own beside it",
                bootstrap.venv_python(comfy_dir),
                app / "comfy-venv-linux" / "bin" / "python")
        rows = {d["id"]: d for d in manager.dependencies(cfg_w)}
        s.check("PyTorch says the old venv is for another system, not Errno 8",
                rows["torch"]["state"] == "missing"
                and "another operating system" in rows["torch"]["detail"]
                and rows["torch"]["action"] == "install",
                rows["torch"]["detail"][:90])
        (app / "comfy-venv" / "bin").mkdir()
        (app / "comfy-venv" / "bin" / "python").write_bytes(b"\x7fELF")
        s.equal("a comfy-venv that also has a native interpreter is used",
                bootstrap.venv_python(comfy_dir),
                app / "comfy-venv" / "bin" / "python")

        portable = Path(tmp) / "ComfyUI_windows_portable"
        (portable / "ComfyUI").mkdir(parents=True)
        (portable / "ComfyUI" / "main.py").write_text("")
        (portable / "python_embeded").mkdir()
        (portable / "python_embeded" / "python.exe").write_bytes(b"MZ\x90\x00")
        s.equal("a portable build's Windows python is passed over on Linux",
                bootstrap.portable_python(portable / "ComfyUI"), None)
        cfg_p = dict(cfg_w, comfy_dir=str(portable / "ComfyUI"), python="",
                     managed=False)
        rows = {d["id"]: d for d in manager.dependencies(cfg_p)}
        s.check("and PyTorch names it instead of 'Install ComfyUI first'",
                rows["torch"]["state"] == "missing"
                and "python_embeded" in rows["torch"]["detail"],
                rows["torch"]["detail"][:90])

        # apt-get from a task with no terminal: sudo would wait for ever
        task = manager.Task("t", "t")
        no = subprocess.CompletedProcess([], 1)
        with patch.object(subprocess, "run", return_value=no), \
                patch.object(manager.shutil, "which",
                             side_effect=lambda n: "/usr/bin/" + n), \
                patch.dict(os.environ, {"DISPLAY": ":0"}):
            s.equal("sudo that needs a password becomes pkexec on a desktop",
                    manager._elevate(["sudo", "apt-get", "install", "-y",
                                      "git"], task),
                    ["pkexec", "apt-get", "install", "-y", "git"])
        with patch.object(subprocess, "run", return_value=no), \
                patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}):
            s.equal("and fails fast as sudo -n without one",
                    manager._elevate(["sudo", "apt-get", "install", "-y",
                                      "git"], task),
                    ["sudo", "-n", "apt-get", "install", "-y", "git"])

    # -- the start-up search: finds ComfyUI anywhere, prefers the weights ----
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def make(rel: str) -> Path:
            c = root / rel
            (c / "models").mkdir(parents=True)
            (c / "main.py").write_text("")
            (c / "folder_paths.py").write_text("")
            return c
        bare = make("a/ComfyUI")
        rich = make("x/y/z/w/ComfyUI")
        make("a/ComfyUI/custom_nodes/inner/ComfyUI")   # never walked into
        (root / "Windows" / "ComfyUI").mkdir(parents=True)
        (root / "Windows" / "ComfyUI" / "main.py").write_text("")
        found = bootstrap.find_comfy_installs([root], max_depth=6, budget=10)
        s.equal("the search finds every ComfyUI, shallowest first",
                found, [bare, rich])
        s.equal("the search stops at the depth limit",
                bootstrap.find_comfy_installs([root], max_depth=3, budget=10),
                [bare])
        wcfg = dict(bootstrap.DEFAULT_CONFIG)
        for m in bootstrap.model_set(wcfg):
            p = bootstrap.model_path(rich / "models", m)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
        s.equal("the install holding the Ideogram 4 weights is the one chosen",
                bootstrap.pick_comfy(found, wcfg), rich)
        # a real ~/ComfyUI on this machine must not answer for the search
        with patch.object(bootstrap, "find_comfy_installs",
                          lambda: [bare, rich]), \
                patch.object(bootstrap, "detect_comfy_dirs", lambda: []):
            lost = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(root / "gone"))
            bootstrap.verify_locations(lost)
            s.equal("a lost ComfyUI is found by the search",
                    lost["comfy_dir"], str(rich))
            s.equal("and its models folder with it",
                    lost["models_dir"], str(rich / "models"))
            s.check("the report says both check out",
                    all("not found" not in line
                        for line in bootstrap.location_report(lost)))
            quiet = dict(bootstrap.DEFAULT_CONFIG,
                         comfy_dir=str(root / "gone"))
            bootstrap.verify_locations(quiet, search=False)
            s.equal("no search when asked not to",
                    quiet["comfy_dir"], str(root / "gone"))
    return s
