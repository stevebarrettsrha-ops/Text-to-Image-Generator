"""Shared plumbing for the suite: reporting, and disposable servers.

Every test gets its own port and its own data directory. Nothing here touches
the library or config of a real install, and two runs cannot collide.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MOCK = Path(__file__).resolve().parent / "mock_comfy.py"
# The suite talks to servers on this machine; a proxy in the environment would
# swallow every request.
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"

sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: E402  (after sys.path, on purpose)

# The four files the default fp8 set expects on disk, straight from the app's
# own model list — so a change to the set cannot leave the tests behind.
WEIGHTS = bootstrap.model_set({"precision": "fp8"})


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
CURRENT: "Suite | None" = None


class Suite:
    """Collects checks so one failure does not hide the rest."""

    def __init__(self, name: str) -> None:
        global CURRENT
        self.name = name
        self.passed = 0
        self.failures: list[str] = []
        CURRENT = self

    def check(self, what: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  ok   {what}" + (f" — {detail}" if detail else ""))
        else:
            self.failures.append(what)
            print(f"  FAIL {what}" + (f" — {detail}" if detail else ""))
        return bool(ok)

    def equal(self, what: str, got, want) -> bool:
        return self.check(what, got == want, f"got {got!r}, wanted {want!r}")

    def skip(self, what: str, why: str) -> None:
        print(f"  --   {what} — skipped: {why}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# disposable servers
# --------------------------------------------------------------------------- #
class Server:
    """A subprocess that is always cleaned up, however the test ends."""

    def __init__(self, argv: list[str], port: int, ready_path: str,
                 env: dict | None = None, cwd: Path = ROOT) -> None:
        self.argv, self.port, self.ready_path = argv, port, ready_path
        self.env, self.cwd = env or {}, cwd
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{port}"
        self.log = Path(tempfile.mkstemp(suffix=".log")[1])

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            self.argv, cwd=str(self.cwd), env={**os.environ, **self.env},
            stdout=self.log.open("w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        for _ in range(160):
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.argv[-1]} died at startup:\n"
                                   f"{self.log.read_text()[-2000:]}")
            try:
                requests.get(self.url + self.ready_path, timeout=2)
                return self
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"{self.argv[-1]} never answered on {self.port}:\n"
                           f"{self.log.read_text()[-2000:]}")

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def tail(self, lines: int = 25) -> str:
        return "\n".join(self.log.read_text().splitlines()[-lines:])


def comfy(**env) -> Server:
    """A stand-in ComfyUI, started by the test rather than by the app —
    an orphan from an earlier launch, or somebody's own install."""
    port = free_port()
    return Server([sys.executable, str(MOCK), str(port)], port,
                  "/system_stats", env=env)


def fake_install(root: Path, stale_first_boot: bool = False) -> Path:
    """A pretend ComfyUI checkout whose main.py serves the mock engine.

    This is what lets the app truly own, stop and restart an engine process in
    tests. Its main.py takes the flags this app launches ComfyUI with, and
    scans its own models folder once, at startup, exactly as ComfyUI does —
    so a file that lands afterwards stays invisible until a restart. With
    stale_first_boot the FIRST launch scans nothing at all, which is the
    orphan an earlier run left behind.
    """
    install = root / "ComfyUI"
    install.mkdir(parents=True, exist_ok=True)
    (install / "main.py").write_text(textwrap.dedent(f"""\
        import argparse, os, pathlib, runpy, sys
        here = pathlib.Path(__file__).parent
        p = argparse.ArgumentParser()
        p.add_argument("--listen"); p.add_argument("--port")
        p.add_argument("--disable-auto-launch", action="store_true")
        a = p.parse_args()
        os.environ["MOCK_MODELS_DIR"] = str(here / "models")
        flag = here / "stale.flag"
        if flag.exists():
            os.environ["MOCK_BLANK_UNETS"] = "1"
            flag.unlink()
            print("model scan found no diffusion models", flush=True)
        else:
            print("model scan read the Ideogram 4 set", flush=True)
        print("Starting server", flush=True)
        sys.argv = ["mock_comfy.py", a.port]
        runpy.run_path({str(MOCK)!r}, run_name="__main__")
    """))
    if stale_first_boot:
        (install / "stale.flag").write_text("first boot is a stale scan")
    fake_weights(install / "models")
    return install


def supervised_comfy() -> Server:
    """A mock engine under a supervisor that respawns it when killed —
    ComfyUI Desktop and launcher scripts behave exactly like this."""
    port = free_port()
    script = Path(tempfile.mkstemp(suffix="_supervisor.py")[1])
    script.write_text(textwrap.dedent(f"""\
        import subprocess, sys, time
        while True:
            p = subprocess.Popen([sys.executable, {str(MOCK)!r}, sys.argv[1]])
            p.wait()
            time.sleep(0.3)
    """))
    return Server([sys.executable, str(script), str(port)], port,
                  "/system_stats")


def foreign_listener(port: int | None = None):
    """Something that is not ComfyUI, holding the port and answering
    /system_stats — a VPN client, a dev server, anything.

    The interpreter runs under a symlink with an innocuous name so that
    /proc/<pid>/cmdline carries no 'python', 'main.py' or 'comfy'. Returns
    (Server, name) or (None, why) where that cannot be arranged.
    """
    port = port or free_port()
    # nothing in the path may read as python/main.py/comfy, or the guard the
    # test is about would be tripped by the test's own scaffolding
    home = Path(tempfile.mkdtemp(prefix="stranger-"))
    alias = home / "vendor-sync-agent"
    script = home / "listen.txt"
    script.write_text(textwrap.dedent("""\
        import sys
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def do_GET(self):
                body = b'{"system": {}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self): self.do_GET()
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
    """))
    try:
        alias.symlink_to(Path(sys.executable).resolve())
        probe = subprocess.run([str(alias), "-c", "print(1)"],
                               capture_output=True, text=True, timeout=30)
        if probe.returncode != 0:
            return None, "the interpreter will not run under another name"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"no aliased interpreter here ({exc})"
    argv = [str(alias), str(script), str(port)]
    if any(k in " ".join(argv).lower()
           for k in ("python", "main.py", "comfy")):
        return None, "the temp path itself looks like ComfyUI"
    return Server(argv, port, "/system_stats"), alias.name


def fake_weights(models_dir: Path) -> None:
    """Drop the weight files where missing_models() looks for them."""
    for item in WEIGHTS:
        path = models_dir / item["folder"] / item["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 16)


def studio(comfy_url: str, data: Path, models_dir: Path | None = None,
           **config) -> Server:
    """Ideogram Studio itself, with its own data folder and config to match."""
    data.mkdir(parents=True, exist_ok=True)
    (data / "config.json").write_text(json.dumps({
        "comfy_url": comfy_url, "comfy_dir": "", "python": "",
        "models_dir": str(models_dir or ""), "managed": False,
        "auto_start_comfy": False, "torch_index": "",
        "precision": "fp8", "setup_complete": True, **config}))
    port = free_port()
    return Server([sys.executable, "server.py"], port, "/api/status",
                  env={"IDEOGRAM_STUDIO_PORT": str(port),
                       "IDEOGRAM_STUDIO_NO_BROWSER": "1",
                       "IDEOGRAM_STUDIO_DATA": str(data)})


class Workspace:
    """A throwaway data directory, removed when the test finishes."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="ig-test-"))
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def wait_for(condition, timeout: float = 30, step: float = 0.5) -> bool:
    """Poll until it is true, rather than sleeping and hoping."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:  # noqa: BLE001  a server mid-restart is expected
            pass
        time.sleep(step)
    return False


def status(app_url: str) -> dict:
    return requests.get(app_url + "/api/status", timeout=10).json()


def engine_log(app_url: str, n: int = 200) -> dict:
    return requests.get(f"{app_url}/api/comfy/log?n={n}", timeout=10).json()
