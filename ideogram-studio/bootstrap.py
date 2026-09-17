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
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
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
        self.steps = {k: {"label": v, "state": "pending", "detail": ""}
                      for k, v in self.STEPS}

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

    def detail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["detail"] = detail

    def finish(self, key: str, detail: str = "") -> None:
        with self._lock:
            self.steps[key]["state"] = "done"
            if detail:
                self.steps[key]["detail"] = detail

    def fail(self, key: str, detail: str) -> None:
        with self._lock:
            self.steps[key]["state"] = "error"
            self.steps[key]["detail"] = detail

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {"running": self.running, "done": self.done,
                    "error": self.error, "step": self.step,
                    "steps": json.loads(json.dumps(self.steps)),
                    "cursor": len(self.lines), "lines": self.lines[since:]}


# --------------------------------------------------------------------------- #
# interpreters
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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


def wait_for_comfy(url: str, timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if comfy_online(url):
            return True
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# pip
# --------------------------------------------------------------------------- #
def pip_install(python: str, args: list[str], log) -> None:
    cmd = [python, "-m", "pip", "install"] + args
    log("$ " + " ".join(cmd[:8]) + (" …" if len(cmd) > 8 else ""))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith(("Collecting", "Downloading", "Installing",
                            "Successfully", "ERROR", "Building", "WARNING: ")):
            log(line[:200])
    if proc.wait() != 0:
        raise RuntimeError("pip install failed — see the log.")


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


def clone_node(node: dict, comfy_dir: Path, log) -> Path:
    """Clone or update one custom node, trying its fallback URL if given."""
    target = comfy_dir / "custom_nodes" / node["dir"]
    if target.exists():
        log(f"Updating {node['label']}")
        _run(["git", "-C", str(target), "pull", "--ff-only"])
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    urls = [node["repo"]] + ([node["fallback"]] if node.get("fallback") else [])
    errors = []
    for url in urls:
        log(f"git clone {url}")
        res = _run(["git", "clone", "--depth", "1", url, str(target)])
        if res.returncode == 0:
            return target
        errors.append((res.stderr or res.stdout)[-300:])
        shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError(f"Could not download {node['label']}: "
                       + " | ".join(errors))


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
                    prog.detail("comfyui", "Downloading ComfyUI…")
                    res = _run(["git", "clone", "--depth", "1", COMFY_REPO,
                                str(comfy_dir)])
                    if res.returncode != 0:
                        raise RuntimeError("git clone failed: " +
                                           (res.stderr or res.stdout)[-600:])
                else:
                    prog.detail("comfyui", "Updating ComfyUI…")
                    _run(["git", "-C", str(comfy_dir), "pull", "--ff-only"])
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
            for node in CUSTOM_NODES:
                if node["id"] == "manager" and not cfg.get("want_manager", True):
                    continue
                prog.detail("nodes", f"Installing {node['label']}…")
                node_paths.append(clone_node(node, comfy_dir, prog.log))
            prog.finish("nodes", ", ".join(n["label"] for n in CUSTOM_NODES
                                           if n["id"] != "manager"
                                           or cfg.get("want_manager", True)))

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
                prog.detail("deps", "Installing PyTorch — the long one…")
                pip_install(str(target), ["--upgrade", "pip", "wheel"], prog.log)
                args = ["torch", "torchvision"]
                idx = torch_index(cfg)
                if idx:
                    args += ["--index-url", idx]
                pip_install(str(target), args, prog.log)
                prog.detail("deps", "Installing ComfyUI requirements…")
                pip_install(str(target),
                            ["-r", str(Path(cfg["comfy_dir"]) / "requirements.txt")],
                            prog.log)
            cfg["python"] = str(target)
            for path in node_paths:
                reqs = path / "requirements.txt"
                if reqs.exists():
                    prog.detail("deps", f"Installing requirements for {path.name}…")
                    pip_install(str(target), ["-r", str(reqs)], prog.log)
                else:
                    prog.log(f"No requirements.txt in {path.name} — skipping.")
            prog.finish("deps", f"Installed into {Path(cfg['python']).name}")

        # 5. models --------------------------------------------------------- #
        prog.begin("models")
        todo = missing_models(models_dir, cfg)
        if not todo:
            prog.finish("models", "Everything is already downloaded")
        else:
            prog.log(f"{len(todo)} file(s) to download "
                     f"({cfg.get('precision', 'fp8')} weights)")
            for item in todo:
                dest = model_path(models_dir, item)

                def on_prog(got, total, speed, eta, _n=item["name"]):
                    prog.detail("models",
                                f"{_n} — {got/1e9:.2f} of {total/1e9:.2f} GB · "
                                f"{speed/1e6:.1f} MB/s · "
                                f"{int(eta//60)}m {int(eta%60)}s left")

                download_file(cfg, cfg.get("hf_repo") or MODEL_REPO,
                              f"{item['folder']}/{item['name']}", dest, on_prog)
                prog.log(f"Downloaded {item['name']}")
            prog.finish("models", "Weights ready")

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
            prog.detail("launch", "Waiting for ComfyUI — the first start is slow…")
            if not wait_for_comfy(url, timeout=900):
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
