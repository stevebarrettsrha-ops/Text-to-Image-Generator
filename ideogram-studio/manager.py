"""
manager.py - getting the machine ready, driven from the front end.

Dependencies: Python, Git, ComfyUI, each custom node, PyTorch, ComfyUI's own
packages, the weights and the running engine — each with an installer.

HuggingFace: browse a repo, download single weight files into the right model
folder, resume, cancel, delete. Repo, token and mirror all come from the page.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import bootstrap
from bootstrap import (APP_DIR, CUSTOM_NODES, MODEL_REPO, PRECISIONS, VAE_FILE,
                       comfy_python, have_git, model_path, model_set,
                       portable_python, venv_python)

DEFAULT_ENDPOINT = bootstrap.HF_BASE
MODEL_FOLDERS = ["diffusion_models", "text_encoders", "vae", "checkpoints",
                 "loras", "clip_vision", "upscale_models"]


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
class Task:
    def __init__(self, kind: str, title: str, meta: dict | None = None) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.title = title
        self.meta = meta or {}
        self.state = "running"
        self.pct = 0.0
        self.detail = ""
        self.lines: list[str] = []
        self.created = time.time()
        self.cancel = False
        self._lock = threading.Lock()

    def log(self, msg: str) -> None:
        with self._lock:
            self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            if len(self.lines) > 1200:
                del self.lines[:600]

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def view(self, since: int = 0) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "title": self.title,
                    "meta": self.meta, "state": self.state,
                    "pct": round(self.pct, 1), "detail": self.detail,
                    "created": self.created, "cursor": len(self.lines),
                    "lines": self.lines[since:]}


class Tasks:
    def __init__(self) -> None:
        self._items: dict[str, Task] = {}
        self._lock = threading.Lock()

    def add(self, task: Task) -> Task:
        with self._lock:
            self._items[task.id] = task
            finished = sorted((t for t in self._items.values()
                               if t.state != "running"), key=lambda t: t.created)
            for old in finished[:-40]:
                self._items.pop(old.id, None)
        return task

    def get(self, task_id: str) -> Task | None:
        return self._items.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._items.values(), key=lambda t: t.created,
                          reverse=True)

    def running(self, kind: str = "") -> list[Task]:
        return [t for t in self.list()
                if t.state == "running" and (not kind or t.kind == kind)]


TASKS = Tasks()


def spawn(kind: str, title: str, fn, meta: dict | None = None) -> Task:
    task = TASKS.add(Task(kind, title, meta))

    def wrapper():
        try:
            fn(task)
            if task.state == "running":
                task.set(state="done", pct=100)
        except Exception as exc:  # noqa: BLE001
            task.log(f"FAILED: {exc}")
            task.set(state="error", detail=str(exc))

    threading.Thread(target=wrapper, daemon=True).start()
    return task


def stream(cmd: list[str], task: Task, keep: tuple[str, ...] = ()) -> int:
    task.log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        if not keep or line.startswith(keep):
            task.log(line[:220])
        if task.cancel:
            proc.terminate()
            task.set(state="cancelled", detail="Cancelled")
            return 1
    return proc.wait()


def _probe(python: str, code: str, timeout: int = 90) -> tuple[int, str]:
    if not python or not Path(python).exists():
        return 1, "no interpreter"
    try:
        out = subprocess.run([python, "-c", code], capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


# --------------------------------------------------------------------------- #
# dependency report
# --------------------------------------------------------------------------- #
def dependencies(cfg: dict, client=None) -> list[dict]:
    items: list[dict] = []

    try:
        py = bootstrap.find_python()
        items.append({"id": "python", "label": "Python 3.10+", "state": "ok",
                      "detail": py, "action": None})
    except Exception as exc:  # noqa: BLE001
        items.append({"id": "python", "label": "Python 3.10+", "state": "missing",
                      "detail": str(exc), "action": None,
                      "hint": "Install it from python.org, then press Recheck."})

    git = shutil.which("git") or ""
    items.append({"id": "git", "label": "Git", "state": "ok" if git else "missing",
                  "detail": git or "Needed to download ComfyUI and the nodes.",
                  "action": None if git else "install"})

    comfy_dir = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else None
    if comfy_dir and (comfy_dir / "main.py").exists():
        items.append({"id": "comfyui", "label": "ComfyUI", "state": "ok",
                      "detail": str(comfy_dir), "action": "update"})
    else:
        items.append({"id": "comfyui", "label": "ComfyUI", "state": "missing",
                      "detail": "Not installed yet.", "action": "install"})

    for node in CUSTOM_NODES:
        installed = bool(comfy_dir and bootstrap.node_installed(comfy_dir, node))
        loaded = None
        if client is not None and node["id"] == "kjnodes":
            loaded = client.has("Ideogram4PromptBuilderKJ")
        if not installed:
            state, detail = "missing", node["why"]
        elif loaded is False:
            state = "warn"
            detail = ("Installed but ComfyUI has not loaded it — check the "
                      "ComfyUI console for IMPORT FAILED, then restart it.")
        else:
            state, detail = "ok", str(comfy_dir / "custom_nodes" / node["dir"])
        items.append({"id": "node:" + node["id"], "label": node["label"],
                      "state": state, "detail": detail,
                      "action": "update" if installed else "install"})

    py_comfy = comfy_python(cfg)
    if py_comfy:
        kind = "portable python_embeded" if "python_embeded" in py_comfy \
            else "virtual environment"
        code, out = _probe(py_comfy,
                           "import torch,json;"
                           "print(json.dumps({'v':torch.__version__,"
                           "'cuda':torch.cuda.is_available(),"
                           "'dev':(torch.cuda.get_device_name(0) "
                           "if torch.cuda.is_available() else ''),"
                           "'vram':(torch.cuda.get_device_properties(0).total_memory "
                           "if torch.cuda.is_available() else 0)}))")
        if code != 0:
            items.append({"id": "torch", "label": "PyTorch", "state": "missing",
                          "detail": f"Not installed in the {kind}.",
                          "action": "install"})
        else:
            import json as _json
            try:
                d = _json.loads(out.splitlines()[-1])
                if d["cuda"]:
                    gb = d["vram"] / 1e9
                    state = "ok" if gb >= 11 else "warn"
                    detail = f"torch {d['v']} — {d['dev']}, {gb:.0f} GB"
                    if gb < 11:
                        detail += " — tight for Ideogram 4; try the nvfp4 or " \
                                  "int8 weights."
                    items.append({"id": "torch", "label": "PyTorch", "state": state,
                                  "detail": detail, "action": "reinstall"})
                else:
                    items.append({"id": "torch", "label": "PyTorch", "state": "warn",
                                  "detail": f"torch {d['v']} — no GPU found. "
                                            "Ideogram 4 on CPU is impractical.",
                                  "action": "reinstall"})
            except Exception:
                items.append({"id": "torch", "label": "PyTorch",
                              "state": "unknown", "detail": out[-140:],
                              "action": "install"})
    else:
        items.append({"id": "torch", "label": "PyTorch", "state": "unknown",
                      "detail": "Install ComfyUI first.", "action": "install"})

    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if models_dir and models_dir.is_dir():
        missing = bootstrap.missing_models(models_dir, cfg)
        if missing:
            items.append({"id": "models", "label": "Ideogram 4 weights",
                          "state": "missing",
                          "detail": "Missing: " + ", ".join(m["name"] for m in missing),
                          "action": "models"})
        else:
            items.append({"id": "models", "label": "Ideogram 4 weights",
                          "state": "ok",
                          "detail": f"All four files present "
                                    f"({cfg.get('precision', 'fp8')}).",
                          "action": "models"})
    else:
        items.append({"id": "models", "label": "Ideogram 4 weights",
                      "state": "unknown", "detail": "Set the models folder first.",
                      "action": "models"})

    online = bootstrap.comfy_online(cfg["comfy_url"])
    items.append({"id": "engine", "label": "Engine",
                  "state": "ok" if online else "missing",
                  "detail": cfg["comfy_url"] if online
                  else "ComfyUI is not answering.",
                  "action": None if online else "start"})
    return items


# --------------------------------------------------------------------------- #
# installers
# --------------------------------------------------------------------------- #
def install_dependency(dep_id: str, cfg: dict, opts: dict) -> Task:
    node = None
    if dep_id.startswith("node:"):
        node = next((n for n in CUSTOM_NODES if n["id"] == dep_id.split(":", 1)[1]),
                    None)
        if not node:
            raise RuntimeError(f"Unknown node '{dep_id}'.")
    titles = {"git": "Install Git", "comfyui": "Install ComfyUI",
              "torch": "Install PyTorch"}
    title = node["label"] if node else titles.get(dep_id, dep_id)

    def run(task: Task) -> None:
        if dep_id == "git":
            _install_git(task)
        elif dep_id == "comfyui":
            _install_comfyui(task, cfg)
        elif dep_id == "torch":
            _install_torch(task, cfg, opts)
        elif node:
            _install_node(task, cfg, node)
        else:
            raise RuntimeError(f"Nothing to install for '{dep_id}'.")

    return spawn("dependency", title, run, {"dep": dep_id})


def _install_git(task: Task) -> None:
    cmds = {"Windows": ["winget", "install", "--id", "Git.Git", "-e", "--source",
                        "winget", "--accept-package-agreements",
                        "--accept-source-agreements"],
            "Darwin": ["brew", "install", "git"],
            "Linux": ["sudo", "apt-get", "install", "-y", "git"]}
    cmd = cmds.get(platform.system())
    if not cmd or not shutil.which(cmd[0]):
        raise RuntimeError("Git has to be installed by hand on this system. "
                           "Install it, then press Recheck.")
    if stream(cmd, task) != 0:
        raise RuntimeError("The Git installer did not finish.")
    task.set(detail="Git installed. Restart Ideogram Studio if it still shows "
                    "as missing.")


def _install_comfyui(task: Task, cfg: dict) -> None:
    if not have_git():
        raise RuntimeError("Install Git first.")
    target = Path(cfg["comfy_dir"]) if cfg.get("comfy_dir") else APP_DIR / "ComfyUI"
    if (target / "main.py").exists():
        task.set(detail="Updating ComfyUI…")
        stream(["git", "-C", str(target), "pull", "--ff-only"], task)
    else:
        task.set(detail="Downloading ComfyUI…")
        if stream(["git", "clone", "--depth", "1", bootstrap.COMFY_REPO,
                   str(target)], task) != 0:
            raise RuntimeError("git clone failed — see the log.")
    cfg["comfy_dir"] = str(target)
    cfg["models_dir"] = cfg.get("models_dir") or str(target / "models")
    bootstrap.save_config(cfg)
    task.set(detail=str(target))


def _install_node(task: Task, cfg: dict, node: dict) -> None:
    comfy_dir = Path(cfg.get("comfy_dir") or "")
    if not (comfy_dir / "main.py").exists():
        raise RuntimeError("Install ComfyUI first.")
    if not have_git():
        raise RuntimeError("Install Git first.")
    task.set(detail=f"Installing {node['label']}…")
    path = bootstrap.clone_node(node, comfy_dir, task.log)
    py = comfy_python(cfg)
    reqs = path / "requirements.txt"
    if py and reqs.exists():
        task.set(detail="Installing its requirements…")
        bootstrap.pip_install(py, ["-r", str(reqs)], task.log)
    task.set(detail="Installed. Restart ComfyUI so it loads the node.")


def _install_torch(task: Task, cfg: dict, opts: dict) -> None:
    comfy_dir = Path(cfg.get("comfy_dir") or "")
    if not (comfy_dir / "main.py").exists():
        raise RuntimeError("Install ComfyUI first.")
    target = portable_python(comfy_dir)
    if target:
        task.log(f"Portable ComfyUI — installing into {target}")
    else:
        vpy = venv_python(comfy_dir)
        if not vpy.exists():
            task.set(detail="Creating the Python environment…")
            base = bootstrap.find_python()
            if stream([base, "-m", "venv", str(comfy_dir.parent / "comfy-venv")],
                      task) != 0:
                raise RuntimeError("Could not create the environment.")
        target = vpy
    cfg["python"] = str(target)
    if opts.get("torch_index") is not None:
        cfg["torch_index"] = opts["torch_index"]
    bootstrap.save_config(cfg)
    index = bootstrap.torch_index(cfg)
    task.set(detail="Installing PyTorch — this is the long one…")
    bootstrap.pip_install(str(target), ["--upgrade", "pip", "wheel"], task.log)
    args = ["torch", "torchvision"]
    if index:
        args += ["--index-url", index]
    bootstrap.pip_install(str(target), args, task.log)
    task.set(detail="Installing ComfyUI requirements…")
    bootstrap.pip_install(str(target),
                          ["-r", str(comfy_dir / "requirements.txt")], task.log)
    task.set(detail="PyTorch installed.")


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
def guess_folder(path: str, repo: str = "") -> str:
    head = path.split("/")[0]
    if head in MODEL_FOLDERS:
        return head
    low = path.lower()
    # The repo name is often the only place the word "lora" appears — the files
    # themselves are usually named after the style.
    if "lora" in low or "lora" in (repo or "").lower():
        return "loras"
    if "vae" in low:
        return "vae"
    if "qwen" in low or "encoder" in low or "clip" in low:
        return "text_encoders"
    return "diffusion_models"


def hf_browse(cfg: dict, repo: str, revision: str = "main") -> dict:
    files = bootstrap.hf_tree(cfg, repo, revision)
    root = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    wanted = {m["name"] for m in model_set(cfg)}
    out = []
    for f in files:
        if f["path"].endswith((".gitattributes", ".md")):
            continue
        folder = guess_folder(f["path"], repo)
        name = Path(f["path"]).name
        out.append({**f, "folder": folder, "name": name,
                    "installed": bool(root and (root / folder / name).exists()),
                    "inset": name in wanted})
    out.sort(key=lambda f: (not f["inset"], -f["size"]))
    return {"repo": repo, "revision": revision, "files": out}


def hf_download(cfg: dict, repo: str, path: str, folder: str = "") -> Task:
    if not cfg.get("models_dir"):
        raise RuntimeError("Set the ComfyUI models folder before downloading.")
    root = Path(cfg["models_dir"])
    folder = folder if folder in MODEL_FOLDERS else guess_folder(path, repo)
    name = Path(path).name
    dest = root / folder / name
    if any(t.meta.get("dest") == str(dest) for t in TASKS.running("download")):
        raise RuntimeError(f"{name} is already downloading.")

    def run(task: Task) -> None:
        task.log(f"{repo}/{path} → {dest}")

        def on_prog(got, total, speed, eta):
            task.set(pct=(got / total * 100) if total else 0,
                     detail=f"{got/1e9:.2f} / {total/1e9:.2f} GB · "
                            f"{speed/1e6:.1f} MB/s · "
                            f"{int(eta//60)}m {int(eta%60)}s left")

        bootstrap.download_file(cfg, repo, path, dest, on_prog,
                                lambda: task.cancel)
        if task.cancel:
            task.set(state="cancelled",
                     detail="Cancelled — the part that downloaded is kept, and "
                            "starting again carries on from there.")
            return
        task.set(pct=100, detail=f"Saved — {dest.stat().st_size/1e9:.2f} GB")

    return spawn("download", name, run, {"repo": repo, "path": path,
                                         "dest": str(dest), "name": name})


def download_set(cfg: dict) -> list[Task]:
    """Queue whatever the chosen precision still needs."""
    root = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if not root:
        raise RuntimeError("Set the ComfyUI models folder first.")
    tasks = []
    for item in bootstrap.missing_models(root, cfg):
        tasks.append(hf_download(cfg, cfg.get("hf_repo") or MODEL_REPO,
                                 f"{item['folder']}/{item['name']}",
                                 item["folder"]))
    return tasks


def local_models(cfg: dict) -> list[dict]:
    root = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    out: list[dict] = []
    if not root or not root.is_dir():
        return out
    for folder in MODEL_FOLDERS:
        d = root / folder
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in (".safetensors", ".ckpt", ".pt",
                                            ".bin", ".gguf", ".part"):
                out.append({"folder": folder, "name": f.name,
                            "size": f.stat().st_size,
                            "partial": f.suffix == ".part"})
    return out


def delete_model(cfg: dict, folder: str, name: str) -> None:
    if not cfg.get("models_dir"):
        raise RuntimeError("No models folder is set.")
    if folder not in MODEL_FOLDERS or "/" in name or "\\" in name or ".." in name:
        raise RuntimeError("That path is not allowed.")
    root = Path(cfg["models_dir"]).resolve()
    target = (root / folder / name).resolve()
    if not target.is_relative_to(root):
        raise RuntimeError("That path is outside the models folder.")
    if not target.exists():
        raise RuntimeError("That file is already gone.")
    target.unlink()


def loras_installed(cfg: dict) -> list[dict]:
    """Whatever is sitting in models/loras, including files dropped in by hand."""
    root = Path(cfg["models_dir"]) / "loras" if cfg.get("models_dir") else None
    out: list[dict] = []
    if not root or not root.is_dir():
        return out
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".ckpt", ".pt", ".bin",
                                        ".part"):
            out.append({"name": str(f.relative_to(root)).replace("\\", "/"),
                        "size": f.stat().st_size, "partial": f.suffix == ".part"})
    return out


def delete_lora(cfg: dict, name: str) -> None:
    if not cfg.get("models_dir"):
        raise RuntimeError("No models folder is set.")
    # LoRA names may carry a subfolder, so a plain separator ban is too strict —
    # but ".." and absolute paths never belong in one.
    parts = Path(name).parts
    if not name or not parts or ".." in parts or Path(name).is_absolute():
        raise RuntimeError("That path is not allowed.")
    root = (Path(cfg["models_dir"]) / "loras").resolve()
    target = (root / name).resolve()
    # A prefix test would accept a sibling folder whose name merely starts with
    # "loras"; is_relative_to compares path components.
    if not target.is_relative_to(root):
        raise RuntimeError("That path is outside the LoRA folder.")
    if not target.exists():
        raise RuntimeError("That file is already gone.")
    target.unlink()


def curated(cfg: dict) -> dict:
    """The weight set for each precision, with what is already on disk."""
    root = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    sets = {}
    for key, p in PRECISIONS.items():
        probe = dict(cfg)
        probe["precision"] = key
        items = []
        for m in model_set(probe):
            items.append({**m, "installed": bool(root)
                          and model_path(root, m).exists()})
        sets[key] = {"label": p["label"], "note": p["note"], "files": items,
                     "complete": all(i["installed"] for i in items)}
    return {"precision": cfg.get("precision", "fp8"), "sets": sets,
            "vae": VAE_FILE}
