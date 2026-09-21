"""
bootstrap.py - first-launch setup for Ideogram Studio.

Steps, in order:
  1. Find a real Python 3.10+ (tested by execution, not PATH lookup).
  2. Find an existing ComfyUI, or clone a managed one into ./ComfyUI.
  3. Clone the custom nodes this workflow needs into ComfyUI/custom_nodes and
     install their requirements with the interpreter ComfyUI itself runs on —
     the portable python_embeded where that is what is there.
  4. Download the Ideogram 4 weights into ComfyUI/models/{diffusion_models,
     text_encoders,vae}.
  5. Start ComfyUI headless and wait for /system_stats.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote

import requests

APP_DIR = Path(__file__).resolve().parent
# Config, gallery and finished images. IDEOGRAM_STUDIO_DATA moves the lot,
# which is what lets the tests run against a throwaway folder instead of the
# library of a real install.
DATA_DIR = Path(os.environ.get("IDEOGRAM_STUDIO_DATA") or (APP_DIR / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
HF_BASE = "https://huggingface.co"
MODEL_REPO = "Comfy-Org/Ideogram-4"

# Custom nodes. ComfyUI-Manager moved org, so the old URL is kept as a fallback
# and tried if the first clone fails.
CUSTOM_NODES = [
    {"id": "manager", "dir": "ComfyUI-Manager",
     "label": "ComfyUI-Manager",
     "repo": "https://github.com/Comfy-Org/ComfyUI-Manager.git",
     "fallback": "https://github.com/ltdrdata/ComfyUI-Manager.git",
     "why": "Installs and updates other nodes from inside ComfyUI."},
    {"id": "kjnodes", "dir": "ComfyUI-KJNodes",
     "label": "ComfyUI-KJNodes",
     "repo": "https://github.com/kijai/ComfyUI-KJNodes.git",
     "fallback": "",
     "why": "Carries Ideogram4PromptBuilderKJ, which turns the prompt fields "
            "and the region layout into Ideogram's structured prompt."},
]

# Weight sets. Pick one precision; the conditional and unconditional models are
# always a matched pair.
PRECISIONS = {
    "fp8": {"label": "fp8 — widest support",
            "cond": "ideogram4_fp8_scaled.safetensors",
            "uncond": "ideogram4_unconditional_fp8_scaled.safetensors",
            "clip": "qwen3vl_8b_fp8_scaled.safetensors",
            "size": 9_280_000_000,
            "note": "Runs on most modern NVIDIA cards. About 24 GB of downloads."},
    "int8": {"label": "int8 — smaller, needs int8 tensor cores",
             "cond": "ideogram4_int8_convrot.safetensors",
             "uncond": "ideogram4_unconditional_int8_convrot.safetensors",
             "clip": "qwen3vl_8b_fp8_scaled.safetensors",
             "size": 0,
             "note": "RTX 30 series and newer."},
    "nvfp4": {"label": "nvfp4 — smallest, RTX 50 series",
              "cond": "ideogram4_nvfp4_mixed.safetensors",
              "uncond": "ideogram4_unconditional_nvfp4_mixed.safetensors",
              "clip": "qwen3vl_8b_nvfp4.safetensors",
              "size": 5_490_000_000,
              "note": "Blackwell cards only. Roughly half the download."},
}

VAE_FILE = "flux2-vae.safetensors"

FOLDER_OF = {"diffusion_models": "diffusion_models",
             "text_encoders": "text_encoders", "vae": "vae"}

DEFAULT_CONFIG = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_dir": "",
    "models_dir": "",
    "python": "",
    "managed": True,
    "auto_start_comfy": True,
    "torch_index": "",
    "hf_token": "",
    "hf_endpoint": HF_BASE,
    "hf_repo": MODEL_REPO,
    "precision": "fp8",
    "want_manager": True,
    "setup_complete": False,
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# model set
# --------------------------------------------------------------------------- #
def model_set(cfg: dict) -> list[dict]:
    """The four files this setup needs, as {folder, name, size, role}."""
    p = PRECISIONS.get(cfg.get("precision") or "fp8", PRECISIONS["fp8"])
    return [
        {"folder": "diffusion_models", "name": p["cond"], "size": p["size"],
         "role": "required", "why": "The Ideogram 4 image model."},
        {"folder": "diffusion_models", "name": p["uncond"], "size": p["size"],
         "role": "required",
         "why": "The unconditional half of the pair — the guider needs both."},
        {"folder": "text_encoders", "name": p["clip"], "size": 0,
         "role": "required", "why": "Qwen3-VL text encoder, reads your prompt."},
        {"folder": "vae", "name": VAE_FILE, "size": 0, "role": "required",
         "why": "Turns the result into an image."},
    ]


def model_path(models_dir: Path, item: dict) -> Path:
    return models_dir / item["folder"] / item["name"]


def missing_models(models_dir: Path, cfg: dict) -> list[dict]:
    return [m for m in model_set(cfg) if not model_path(models_dir, m).exists()]


def node_installed(comfy_dir: Path, node: dict) -> bool:
    return (comfy_dir / "custom_nodes" / node["dir"]).is_dir()


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #
class Progress:
    STEPS = [
        ("python", "Check Python"),
        ("comfyui", "Install ComfyUI"),
        ("nodes", "Install the custom nodes"),
        ("deps", "Install dependencies"),
        ("models", "Download the Ideogram 4 models"),
        ("launch", "Start ComfyUI"),
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.done = False
        self.error: str | None = None
        self.step = ""
        self.steps = {k: {"key": k, "label": v, "state": "pending",
                          "detail": "", "pct": None} for k, v in self.STEPS}

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 4000:
                del self.lines[:2000]
        print(f"[setup] {msg}", flush=True)

    def begin(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.step = key
            self.steps[key]["state"] = "running"
            self.steps[key]["detail"] = detail
            self.steps[key]["pct"] = None

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def track(self, key: str, pct: float | None, detail: str = "") -> None:
        """Move a step's bar. `pct` None means running with no number yet —
        the front end shows an indeterminate bar rather than a fake 0%."""
        with self._lock:
            self.steps[key]["pct"] = (None if pct is None
                                      else round(max(0.0, min(100.0, pct)), 1))
            if detail:
                self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            self.steps[key]["pct"] = None
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["pct"] = None
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            # A list, not a dict: jsonify sorts dict keys, which would hand the
            # page the six steps in alphabetical order instead of run order.
            steps = [dict(self.steps[k]) for k, _ in self.STEPS]
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step, "steps": steps,
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# human-readable numbers
# --------------------------------------------------------------------------- #
def fmt_size(n: float) -> str:
    n = float(n or 0)
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB" if n < 1e8 else f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


def fmt_eta(seconds: float) -> str:
    s = int(max(seconds or 0, 0))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m left"
    if s >= 60:
        return f"{s // 60}m {s % 60}s left"
    return f"{s}s left" if s else "almost there"


def fmt_transfer(got: float, total: float, speed: float, eta: float) -> str:
    """One line of download state: how much, how fast, how much longer."""
    bits = [f"{fmt_size(got)} of {fmt_size(total)}" if total
            else f"{fmt_size(got)} so far"]
    if speed > 0:
        bits.append(f"{speed / 1e6:.1f} MB/s")
    if total and speed > 0:
        bits.append(fmt_eta(eta))
    return " \u00b7 ".join(bits)


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _stream(cmd: list[str], on_line, cwd: str | None = None,
            env: dict | None = None, should_cancel=None) -> int:
    r"""Run `cmd` and hand every line of its output to `on_line` as it appears.

    Splits on carriage returns as well as newlines: git writes its progress by
    rewriting one line with \r, so a plain line iterator would hold all of it
    back until the clone finished — which is exactly the silence this is meant
    to fill. Reads with read1() so a partial block is delivered straight away.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=cwd, env=env)
    assert proc.stdout
    buf = b""
    while True:
        block = proc.stdout.read1(8192)
        if not block:
            break
        buf += block
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            text = raw.decode("utf-8", "replace").strip()
            if text:
                on_line(text)
        if should_cancel and should_cancel():
            proc.terminate()
            break
    if buf.strip():
        on_line(buf.decode("utf-8", "replace").strip())
    return proc.wait()


# git reports each phase as its own 0-100%; "Receiving objects" is the download.
GIT_PHASE = re.compile(r"(Counting objects|Compressing objects|Receiving objects"
                       r"|Resolving deltas|Updating files):\s+(\d+)%")


def git_run(cmd: list[str], log, on_pct=None,
            should_cancel=None) -> tuple[int, str]:
    """A git command with its progress forwarded. Returns (code, last output)."""
    tail: list[str] = []

    def line(text: str) -> None:
        m = GIT_PHASE.search(text)
        if m:
            if on_pct:
                on_pct(m.group(1), float(m.group(2)))
            return
        tail.append(text)
        log(text[:200])

    # C locale: the phase names below are what git prints in English, and a
    # translated git would otherwise report no progress at all.
    code = _stream(cmd, line, env=dict(os.environ, LC_ALL="C"),
                   should_cancel=should_cancel)
    return code, "\n".join(tail[-8:])


def git_clone(url: str, target: Path, log, on_pct=None, depth: int = 1,
              should_cancel=None) -> tuple[int, str]:
    return git_run(["git", "clone", "--depth", str(depth), "--progress",
                    url, str(target)], log, on_pct, should_cancel)


def find_python(prog: Progress | None = None) -> str:
    candidates: list[list[str]] = [[sys.executable]]
    if platform.system() == "Windows":
        candidates += [["py", "-3.12"], ["py", "-3.11"], ["py", "-3.10"],
                       ["py", "-3"], ["python"]]
    else:
        candidates += [["python3.12"], ["python3.11"], ["python3.10"],
                       ["python3"], ["python"]]
    for cand in candidates:
        try:
            out = _run(cand + ["-c", "import sys;print(sys.executable);"
                                     "print('%d.%d' % sys.version_info[:2])"],
                       timeout=25)
        except Exception:
            continue
        if out.returncode != 0:
            continue
        parts = [p.strip() for p in out.stdout.strip().splitlines() if p.strip()]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            major, minor = (int(x) for x in parts[1].split("."))
        except ValueError:
            continue
        if (major, minor) >= (3, 10):
            if prog:
                prog.log(f"Using Python {parts[1]} at {parts[0]}")
            return parts[0]
    raise RuntimeError("No Python 3.10 or newer found. Install it from "
                       "python.org, tick 'Add to PATH', and run setup again.")


def portable_python(comfy_dir: Path) -> Path | None:
    for base in (comfy_dir.parent, comfy_dir):
        cand = base / "python_embeded" / "python.exe"
        if cand.exists():
            return cand
    return None


def venv_python(comfy_dir: Path) -> Path:
    venv = comfy_dir.parent / "comfy-venv"
    return venv / ("Scripts/python.exe" if platform.system() == "Windows"
                   else "bin/python")


def comfy_python(cfg: dict) -> str:
    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir:
        p = portable_python(comfy_dir)
        if p:
            return str(p)
        v = venv_python(comfy_dir)
        if v.exists():
            return str(v)
    return cfg.get("python") or ""


def have_git() -> bool:
    return shutil.which("git") is not None


def detect_comfy_dirs() -> list[str]:
    home = Path.home()
    cands = [APP_DIR / "ComfyUI", home / "ComfyUI",
             home / "Documents" / "ComfyUI", home / "Desktop" / "ComfyUI",
             Path("C:/ComfyUI"), Path("C:/ComfyUI_windows_portable/ComfyUI"),
             Path("D:/ComfyUI"), Path("D:/ComfyUI_windows_portable/ComfyUI")]
    appdata = os.environ.get("APPDATA")
    local = os.environ.get("LOCALAPPDATA")
    if appdata:
        cands.append(Path(appdata) / "ComfyUI")
    if local:
        cands.append(Path(local) / "Programs" / "@comfyorgcomfyui-electron"
                     / "resources" / "ComfyUI")
    out, seen = [], set()
    for c in cands:
        try:
            if ((c / "main.py").exists() or (c / "models").is_dir()) \
                    and str(c) not in seen:
                seen.add(str(c))
                out.append(str(c))
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def hf_headers(cfg: dict) -> dict:
    token = (cfg.get("hf_token") or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_endpoint(cfg: dict) -> str:
    return (cfg.get("hf_endpoint") or HF_BASE).rstrip("/")


def hf_tree(cfg: dict, repo: str, revision: str = "main") -> list[dict]:
    base = hf_endpoint(cfg)
    last = ""
    for kind in ("models", "datasets"):
        url = f"{base}/api/{kind}/{repo}/tree/{revision}?recursive=1"
        try:
            r = requests.get(url, headers=hf_headers(cfg), timeout=30)
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
            continue
        if r.status_code == 401:
            raise RuntimeError("This repo needs a HuggingFace token. Add one on "
                               "the Models page, then try again.")
        if r.status_code == 403:
            raise RuntimeError("Your token cannot read this repo. Accept the "
                               "model licence on huggingface.co first — "
                               "Ideogram 4 is under a non-commercial agreement.")
        if r.status_code == 404:
            continue
        r.raise_for_status()
        files = []
        for e in r.json():
            if e.get("type") != "file":
                continue
            size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
            files.append({"path": e["path"], "size": size})
        return files
    raise RuntimeError(f"Could not find '{repo}' on {base}. "
                       + (last or "Check the spelling, or add a token."))


def download_file(cfg: dict, repo: str, path: str, dest: Path,
                  on_progress=None, should_cancel=None,
                  revision: str = "main") -> None:
    """Resumable: .part file, Range resume, atomic move."""
    url = f"{hf_endpoint(cfg)}/{repo}/resolve/{revision}/{path}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(hf_headers(cfg))
    if have:
        headers["Range"] = f"bytes={have}-"

    with requests.get(url, headers=headers, stream=True, timeout=60,
                      allow_redirects=True) as r:
        if r.status_code == 416:
            part.replace(dest)
            return
        if r.status_code in (401, 403):
            raise RuntimeError("HuggingFace refused the download. Accept the "
                               "Ideogram 4 licence on the model page, then add "
                               "a token on the Models page.")
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        if mode == "wb":
            have = 0          # the server ignored Range — starting over
        # Work out the size after that reset: on a 206 Content-Length is what
        # is left, on a 200 it is the whole file. Adding `have` to a restart
        # would double-count the bytes already on disk.
        total = int(r.headers.get("Content-Length", 0)) + have
        got, last, started = have, 0.0, time.time()
        with open(part, mode) as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if should_cancel and should_cancel():
                    return
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if on_progress and now - last > 0.6:
                    last = now
                    speed = (got - have) / max(now - started, .1)
                    eta = (total - got) / speed if speed > 0 and total else 0
                    on_progress(got, total, speed, eta)
    # A cut-off transfer just ends the iterator — no exception. Promoting a
    # short file to the real name would leave a model that looks installed and
    # fails to load, so stop here and keep the .part for the next resume.
    if total and got < total:
        raise RuntimeError(
            f"The download stopped early — {got/1e9:.2f} of {total/1e9:.2f} GB "
            "arrived. What came through is kept; start it again and it carries "
            "on from there.")
    part.replace(dest)


# --------------------------------------------------------------------------- #
# ComfyUI process
# --------------------------------------------------------------------------- #
class ComfyProcess:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def note(self, msg: str) -> None:
        """An app-side line in the engine console — what the app is doing TO
        the engine belongs next to what the engine itself says."""
        with self._lock:
            self.lines.append(f"[Ideogram Studio] {msg}")
            if len(self.lines) > 2000:
                del self.lines[:1000]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, python: str, comfy_dir: Path, port: int,
              prog: Progress) -> None:
        if self.alive():
            return
        cmd = [python, "main.py", "--listen", "127.0.0.1", "--port", str(port),
               "--disable-auto-launch"]
        prog.log("Launching ComfyUI: " + " ".join(cmd))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            if platform.system() == "Windows" else 0
        self.proc = subprocess.Popen(cmd, cwd=str(comfy_dir),
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, creationflags=flags)
        threading.Thread(target=self._pump, args=(prog,), daemon=True).start()

    def _pump(self, prog: Progress) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.rstrip()
            with self._lock:
                self.lines.append(line)
                if len(self.lines) > 2000:
                    del self.lines[:1000]
            if any(k in line for k in ("Error", "Traceback", "error:",
                                       "IMPORT FAILED", "Starting server")):
                prog.log(f"ComfyUI: {line}")

    def tail(self, n: int = 40) -> list[str]:
        with self._lock:
            return self.lines[-n:]

    def stop(self) -> None:
        if self.alive():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def comfy_online(url: str) -> bool:
    try:
        return requests.get(f"{url}/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def _pids_from_proc_net(port: int) -> list[int]:
    """Linux, no tools needed: the socket inode from /proc/net/tcp*, then the
    process whose fd table holds it."""
    inodes = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local, state, inode = parts[1], parts[3], parts[9]
            if state == "0A" and local.rsplit(":", 1)[-1] == f"{port:04X}":
                inodes.add(inode)
    pids = set()
    if not inodes:
        return []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if any(f"socket:[{i}]" == target for i in inodes):
                    pids.add(int(proc.name))
                    break
        except OSError:
            continue
    return sorted(pids)


def port_pids(port: int) -> list[int]:
    """Whoever is listening on the port."""
    if platform.system() == "Windows":
        pids = set()
        try:
            out = _run(["netstat", "-ano", "-p", "TCP"], timeout=25).stdout
        except Exception:
            return []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" \
                    and parts[3] == "LISTENING" \
                    and parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
        return sorted(pids)
    found = _pids_from_proc_net(port)
    if found:
        return found
    if shutil.which("lsof"):
        try:
            out = _run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                       timeout=25).stdout
            return sorted({int(t) for t in out.split() if t.strip().isdigit()})
        except Exception:
            pass
    return []


def pid_cmdline(pid: int) -> str:
    try:
        if platform.system() == "Windows":
            out = _run(["wmic", "process", "where", f"processid={pid}",
                        "get", "commandline"], timeout=25).stdout
            lines = [ln.strip() for ln in out.splitlines()
                     if ln.strip() and "CommandLine" not in ln]
            return lines[0] if lines else ""
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists():
            return cmd.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()
        return _run(["ps", "-p", str(pid), "-o", "command="],
                    timeout=25).stdout.strip()
    except Exception:
        return ""


def pid_alive(pid: int) -> bool:
    """Return whether *pid* can still execute code.

    ``kill(pid, 0)`` also succeeds for a zombie.  That distinction matters in
    containers whose PID 1 does not promptly reap orphaned children: waiting
    for such a process to disappear makes a successful stop look like a
    timeout, followed by a pointless SIGKILL.  Linux exposes the state in
    ``/proc``; other Unix systems retain the traditional signal probe.
    """
    # Process ids from the port-discovery paths are positive.  Guard this
    # public helper as well: on POSIX, 0 and negative values target process
    # groups rather than one process.
    if pid <= 0:
        return False
    if platform.system() == "Linux":
        try:
            # The command name is parenthesised and may contain spaces, so the
            # state is the first field after the final closing parenthesis.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)
            if len(fields) == 2 and fields[1].split()[0] == "Z":
                return False
        except FileNotFoundError:
            # Linux normally has procfs mounted, in which case a missing pid
            # entry proves the process is gone.  Minimal/chroot environments
            # may omit procfs entirely, so retain the signal-probe fallback.
            if Path("/proc").is_dir():
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_pid(pid: int) -> str:
    """Stop a process: politely first, firmly if it lingers. Returns what the
    system said about it, so a refusal (access denied, already gone) can be
    shown instead of guessed at."""
    if platform.system() == "Windows":
        try:
            out = _run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=30)
            return (out.stdout or out.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            return str(exc)
    if not pid_alive(pid):
        return "already gone"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already gone"
    except PermissionError:
        return "access denied"
    for _ in range(25):
        time.sleep(0.2)
        if not pid_alive(pid):
            return "stopped"
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "stopped"
    except PermissionError:
        return "access denied"
    # A killed orphan can remain as a zombie until its parent reaps it.  It is
    # already stopped, but retain "sent SIGKILL" so the console accurately
    # records that the forceful path was required.
    return "sent SIGKILL"


def comfy_stats(url: str) -> dict | None:
    """What is actually answering on the address — argv says which install."""
    try:
        r = requests.get(f"{url}/system_stats", timeout=3)
        if r.status_code == 200:
            return r.json().get("system") or {}
    except Exception:
        pass
    return None


def wait_for_comfy(url: str, timeout: int = 900, on_wait=None) -> bool:
    """Poll until ComfyUI answers. `on_wait(elapsed, timeout)` runs each pass —
    there is no percentage to give here, only how long it has been waiting."""
    started = time.time()
    deadline = started + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        if on_wait:
            on_wait(time.time() - started, timeout)
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip
# --------------------------------------------------------------------------- #
# pip's own progress bar is a terminal animation and vanishes when its output
# is a pipe, which is why a 2.4 GB torch wheel looks like a hang. `--progress-bar
# raw` makes it print "Progress <done> of <total>" lines instead, which survive
# the pipe. Older pips do not have it, so ask before using it.
PIP_RAW = re.compile(r"^Progress (\d+) of (\d+)$")
PIP_GET = re.compile(r"^\s*(?:Downloading|Using cached)\s+(\S+)")
_PIP_RAW_OK: dict[str, bool] = {}


def pip_has_raw_progress(python: str) -> bool:
    if python not in _PIP_RAW_OK:
        ok = False
        try:
            out = _run([python, "-m", "pip", "install", "--help"], timeout=60)
            at = out.stdout.find("--progress-bar")
            ok = at >= 0 and "raw" in out.stdout[at:at + 300]
        except Exception:  # noqa: BLE001
            ok = False
        _PIP_RAW_OK[python] = ok
    return _PIP_RAW_OK[python]


def pip_install(python: str, args: list[str], log, on_pct=None,
                should_cancel=None) -> None:
    """Install with pip. `on_pct(pct|None, detail)` is called as it downloads."""
    cmd = [python, "-m", "pip", "install"]
    if on_pct and pip_has_raw_progress(python):
        cmd += ["--progress-bar", "raw"]
    cmd += args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))

    # Per-wheel transfer state: pip restarts the counter for every file.
    cur = {"name": "", "base": 0, "started": 0.0, "last": 0.0}

    def line(text: str) -> None:
        m = PIP_RAW.match(text)
        if m:
            if not on_pct:
                return
            got, total = int(m.group(1)), int(m.group(2))
            now = time.time()
            if got < cur["base"] or not cur["started"]:
                cur["base"], cur["started"] = got, now       # a new file
            if now - cur["last"] < 0.4 and not (total and got >= total):
                return
            cur["last"] = now
            speed = (got - cur["base"]) / max(now - cur["started"], .1)
            eta = (total - got) / speed if speed > 0 and total else 0
            label = cur["name"] or "package"
            on_pct((got / total * 100) if total else None,
                   f"{label} — {fmt_transfer(got, total, speed, eta)}")
            return
        m = PIP_GET.match(text)
        if m:
            cur.update(name=unquote(m.group(1).rsplit("/", 1)[-1])[:60],
                       base=0, started=0.0, last=0.0)
        if text.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(text[:200])
            if on_pct and text.startswith(("Installing", "Building")):
                # Unpacking and byte-compiling: no byte count to report, and
                # torch takes minutes over it, so say what is happening.
                on_pct(None, text[:120])

    if _stream(cmd, line, should_cancel=should_cancel) != 0:
        raise RuntimeError("pip install failed — see the log.")
    if "pip" in args:
        # pip just upgraded itself, so whether it can report progress may have
        # changed. The very first install in a new venv is that upgrade.
        _PIP_RAW_OK.pop(python, None)


def torch_index(cfg: dict) -> str:
    if cfg.get("torch_index"):
        return cfg["torch_index"]
    if platform.system() == "Darwin":
        return ""
    if shutil.which("nvidia-smi"):
        try:
            if _run(["nvidia-smi"], timeout=20).returncode == 0:
                return "https://download.pytorch.org/whl/cu128"
        except Exception:
            pass
    return "https://download.pytorch.org/whl/cpu"


def clone_node(node: dict, comfy_dir: Path, log, on_pct=None,
               should_cancel=None) -> Path:
    """Clone or update one custom node, trying its fallback URL if given."""
    target = comfy_dir / "custom_nodes" / node["dir"]
    if target.exists():
        log(f"Updating {node['label']}")
        git_run(["git", "-C", str(target), "pull", "--ff-only", "--progress"],
                log, on_pct, should_cancel)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [node["repo"]] + ([node["fallback"]] if node.get("fallback") else [])
    errors = []
    for url in urls:
        log(f"git clone {url}")
        code, out = git_clone(url, target, log, on_pct,
                              should_cancel=should_cancel)
        if code == 0:
            return target
        errors.append(out[-300:])
        shutil.rmtree(target, ignore_errors=True)
        if should_cancel and should_cancel():
            raise RuntimeError("Cancelled.")
    raise RuntimeError(f"Could not download {node['label']}: "
                       + " | ".join(errors))


# --------------------------------------------------------------------------- #
# progress adapters
# --------------------------------------------------------------------------- #
# Each git phase is its own 0-100%, so stack them into one bar that only ever
# moves forwards. Receiving objects is the transfer and takes nearly all of it.
GIT_WEIGHT = {"Counting objects": (0.00, 0.02), "Compressing objects": (0.02, 0.03),
              "Receiving objects": (0.05, 0.90), "Resolving deltas": (0.95, 0.04),
              "Updating files": (0.95, 0.05)}


def _git_pct(prog: Progress, key: str, head: str = "",
             base: float = 0.0, span: float = 100.0):
    """A git on_pct callback that drives `key`'s bar between base and base+span."""
    def on_pct(phase: str, pct: float) -> None:
        start, width = GIT_WEIGHT.get(phase, (0.0, 0.0))
        line = f"{phase} — {pct:.0f}%"
        prog.track(key, base + (start + width * pct / 100) * span,
                   f"{head} · {line}" if head else line)
    return on_pct


def _pip_pct(prog: Progress, head: str, key: str = "deps"):
    """A pip on_pct callback: pct is None while pip is not transferring bytes."""
    def on_pct(pct: float | None, detail: str) -> None:
        prog.track(key, pct, f"{head} · {detail}")
    return on_pct


# --------------------------------------------------------------------------- #
# setup run
# --------------------------------------------------------------------------- #
def run_setup(cfg: dict, prog: Progress, comfy: ComfyProcess,
              chosen_dir: str = "", mode: str = "auto") -> None:
    prog.running = True
    prog.done = False
    prog.error = None
    try:
        # 1. python -------------------------------------------------------- #
        prog.begin("python")
        if mode == "external":
            prog.finish("python", "Not needed — you run ComfyUI yourself")
            py = cfg.get("python") or sys.executable
        else:
            py = find_python(prog)
            cfg["python"] = py
            prog.finish("python", py)

        # 2. comfyui ------------------------------------------------------- #
        prog.begin("comfyui")
        comfy_dir = None
        if mode == "external":
            if not comfy_online(cfg["comfy_url"]):
                raise RuntimeError(f"Nothing is answering at {cfg['comfy_url']}. "
                                   "Start ComfyUI first, or let Ideogram Studio "
                                   "install its own.")
            if not cfg.get("models_dir"):
                raise RuntimeError("Set the ComfyUI models folder in Settings so "
                                   "the weights land in the right place.")
            cfg["managed"] = False
            if cfg.get("comfy_dir"):
                comfy_dir = Path(cfg["comfy_dir"])
            prog.finish("comfyui", cfg["comfy_url"])
        else:
            if chosen_dir:
                comfy_dir = Path(chosen_dir)
                cfg["managed"] = False
                prog.log(f"Using existing ComfyUI at {comfy_dir}")
            else:
                comfy_dir = APP_DIR / "ComfyUI"
                cfg["managed"] = True
                if not (comfy_dir / "main.py").exists():
                    if not have_git():
                        raise RuntimeError(
                            "Git is not installed, so ComfyUI cannot be "
                            "downloaded. Install Git from the Engine page, or "
                            "point Ideogram Studio at an existing ComfyUI.")
                    prog.track("comfyui", None, "Downloading ComfyUI…")
                    code, out = git_clone(COMFY_REPO, comfy_dir, prog.log,
                                          _git_pct(prog, "comfyui"))
                    if code != 0:
                        raise RuntimeError("git clone failed: " + out[-600:])
                else:
                    prog.track("comfyui", None, "Updating ComfyUI…")
                    git_run(["git", "-C", str(comfy_dir), "pull", "--ff-only",
                             "--progress"], prog.log, _git_pct(prog, "comfyui"))
            if not (comfy_dir / "main.py").exists():
                raise RuntimeError(f"No main.py in {comfy_dir} — that folder is "
                                   "not a ComfyUI install.")
            cfg["comfy_dir"] = str(comfy_dir)
            cfg["models_dir"] = str(comfy_dir / "models")
            prog.finish("comfyui", str(comfy_dir))

        models_dir = Path(cfg["models_dir"])

        # 3. custom nodes --------------------------------------------------- #
        prog.begin("nodes")
        node_paths: list[Path] = []
        if comfy_dir is None:
            prog.finish("nodes", "Install the nodes in your own ComfyUI")
        else:
            if not have_git():
                raise RuntimeError("Git is needed to install the custom nodes.")
            wanted = [n for n in CUSTOM_NODES
                      if n["id"] != "manager" or cfg.get("want_manager", True)]
            for i, node in enumerate(wanted, 1):
                head = f"{node['label']} ({i} of {len(wanted)})"
                prog.track("nodes", (i - 1) / len(wanted) * 100,
                           f"Installing {head}…")
                node_paths.append(clone_node(
                    node, comfy_dir, prog.log,
                    _git_pct(prog, "nodes", head, (i - 1) / len(wanted) * 100,
                             1 / len(wanted) * 100)))
            prog.finish("nodes", ", ".join(n["label"] for n in wanted))

        # 4. dependencies --------------------------------------------------- #
        prog.begin("deps")
        if mode == "external":
            prog.finish("deps", "Handled by your own ComfyUI install")
        else:
            target = portable_python(Path(cfg["comfy_dir"]))
            if target:
                prog.log(f"Portable ComfyUI detected — installing into {target}")
            else:
                vpy = venv_python(Path(cfg["comfy_dir"]))
                if not vpy.exists():
                    prog.detail("deps", "Creating the Python environment…")
                    res = _run([py, "-m", "venv",
                                str(Path(cfg["comfy_dir"]).parent / "comfy-venv")])
                    if res.returncode != 0:
                        raise RuntimeError("venv creation failed: " +
                                           (res.stderr or res.stdout)[-600:])
                target = vpy
                prog.track("deps", None, "Updating pip…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"],
                            prog.log, _pip_pct(prog, "pip and wheel"))
                args = ["torch", "torchvision"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                prog.track("deps", None, "Installing PyTorch — the long one…")
                pip_install(str(target), args, prog.log,
                            _pip_pct(prog, "PyTorch"))
                prog.track("deps", None, "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(Path(cfg["comfy_dir"]) / "requirements.txt")],
                            prog.log, _pip_pct(prog, "ComfyUI requirements"))
            cfg["python"] = str(target)
            for path in node_paths:
                reqs = path / "requirements.txt"
                if reqs.exists():
                    prog.track("deps", None,
                               f"Installing requirements for {path.name}…")
                    pip_install(str(target), ["-r", str(reqs)], prog.log,
                                _pip_pct(prog, path.name))
                else:
                    prog.log(f"No requirements.txt in {path.name} — skipping.")
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        # 5. models --------------------------------------------------------- #
        prog.begin("models")
        todo = missing_models(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            repo = cfg.get("hf_repo") or MODEL_REPO
            # Ask the repo for the real sizes so the bar can cover the whole
            # set. The hard-coded sizes are rough and some are zero, and a bar
            # that only knows the current file jumps back to 0% four times.
            sizes: dict[str, int] = {}
            try:
                prog.track("models", None, "Checking what is on the repo…")
                sizes = {f["path"]: f["size"] for f in hf_tree(cfg, repo)}
            except Exception as exc:  # noqa: BLE001
                prog.log(f"Could not read the file list ({exc}). The bar will "
                         "follow one file at a time instead of the whole set.")
            plan = []
            for item in todo:
                path = f"{item['folder']}/{item['name']}"
                plan.append((item, path,
                             int(sizes.get(path) or item.get("size") or 0)))
            grand = sum(s for _, _, s in plan)
            prog.log(f"{len(plan)} file(s) to download "
                     f"({cfg.get('precision', 'fp8')} weights"
                     + (f", {fmt_size(grand)}" if grand else "") + ")")
            done_bytes = 0
            for i, (item, path, size) in enumerate(plan, 1):
                dest = model_path(models_dir, item)
                head = f"{item['name']} ({i} of {len(plan)})"

                def on_prog(got, total, speed, eta, _head=head):
                    whole = ((done_bytes + got) / grand * 100) if grand else (
                        (got / total * 100) if total else None)
                    prog.track("models", whole,
                               f"{_head} — "
                               + fmt_transfer(got, total, speed, eta))

                prog.track("models",
                           (done_bytes / grand * 100) if grand else None,
                           f"{head} — starting…")
                download_file(cfg, repo, path, dest, on_prog)
                done_bytes += size or (dest.stat().st_size if dest.exists() else 0)
                prog.log(f"Downloaded {item['name']}")
            prog.finish("models", f"{len(plan)} file(s) ready"
                        + (f" · {fmt_size(grand)}" if grand else ""))

        # 6. launch --------------------------------------------------------- #
        prog.begin("launch")
        url = cfg["comfy_url"]
        if mode == "external" or not cfg.get("auto_start_comfy", True):
            if not comfy_online(url):
                raise RuntimeError(f"ComfyUI is not answering at {url}.")
        elif comfy_online(url):
            prog.log("ComfyUI is already running — restart it so it picks up "
                     "the new nodes.")
        else:
            port = int(url.rsplit(":", 1)[-1])
            comfy.start(cfg["python"], Path(cfg["comfy_dir"]), port, prog)

            def waiting(elapsed: float, limit: int) -> None:
                s = int(elapsed)
                been = f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"
                prog.track("launch", None, "Waiting for ComfyUI — "
                           f"{been} so far. The first start is slow.")

            prog.track("launch", None, "Waiting for ComfyUI…")
            if not wait_for_comfy(url, timeout=900, on_wait=waiting):
                raise RuntimeError("ComfyUI did not start within 15 minutes.\n"
                                   + "\n".join(comfy.tail(25)))
        prog.finish("launch", url)

        cfg["setup_complete"] = True
        save_config(cfg)
        prog.done = True
        prog.log("Setup complete. Ideogram Studio is ready.")
    except Exception as exc:  # noqa: BLE001
        prog.error = str(exc)
        if prog.step:
            prog.fail(prog.step, str(exc))
        prog.log(f"FAILED: {exc}")
    finally:
        prog.running = False
