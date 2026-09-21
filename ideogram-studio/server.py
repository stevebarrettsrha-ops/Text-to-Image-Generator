"""
server.py - Ideogram Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7802)
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

import requests

from flask import (Flask, Response, jsonify, request, send_file,
                   send_from_directory)

import bootstrap
import manager
from bootstrap import (APP_DIR, ComfyProcess, Progress, comfy_online,
                       detect_comfy_dirs, load_config, save_config)
from comfy import ComfyClient, ComfyError

DATA_DIR = bootstrap.DATA_DIR          # honours IDEOGRAM_STUDIO_DATA
IMAGES_DIR = DATA_DIR / "images"
GALLERY_PATH = DATA_DIR / "gallery.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("IDEOGRAM_STUDIO_PORT", "7802"))

# Which of ComfyUI's lists decides whether this app can generate at all, and
# what to look for in it. resolve_models() reads the diffusion-model list
# (UNETLoader.unet_name) first and refuses without it, and every weight the
# Models page downloads for the image model — every precision, both halves of
# the pair — is named ideogram4_*.safetensors. An empty-of-ideogram4 list next
# to a full models folder therefore means one thing: the engine scanned before
# the weights landed. The text encoder and VAE lists are not the test; a run
# fails on the diffusion models first.
MODEL_MARKER = "ideogram4"
# The node the graph cannot be built without, and the pack markers worth
# checking: ComfyUI-Manager registers no node class, KJNodes carries the
# prompt builder.
CORE_NODE = "DualModelGuider"
NODE_MARKERS = {"kjnodes": "Ideogram4PromptBuilderKJ"}

app = Flask(__name__, static_folder=None)
# Flask sorts JSON object keys by default, which would reorder anything
# the page shows in the order the server put it in.
app.json.sort_keys = False


@app.before_request
def block_cross_site():
    """This API installs software, downloads weights and deletes files, and it
    has no login — anything that can reach it can drive it. A page on any site
    can send a 'simple' POST to 127.0.0.1 with no preflight to stop it, so
    refuse writes that a browser tells us came from somewhere else. Requests
    with no Origin (curl, scripts) are left alone."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return None
    origin = request.headers.get("Origin")
    if origin and urlparse(origin).netloc != request.host:
        return jsonify({"error": "Refused: that request came from another "
                                 "site."}), 403
    return None

cfg = load_config()
progress = Progress()
comfy_proc = ComfyProcess()
client = ComfyClient(cfg["comfy_url"])

JOB_TTL = 1800          # finished jobs stay visible this long, then go
MAX_JOBS = 200

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gallery_lock = threading.Lock()
ws_progress: dict[str, dict] = {}


def prune_jobs() -> None:
    """Caller holds jobs_lock. Finished jobs linger so the feed can show the
    last result, then are dropped — otherwise the dict grows for the life of
    the process. Running jobs are never touched."""
    now = time.time()
    for job in [j for j in jobs.values()
                if j["status"] != "running" and now - j["created"] > JOB_TTL]:
        jobs.pop(job["id"], None)
    if len(jobs) > MAX_JOBS:
        done = sorted((j for j in jobs.values() if j["status"] != "running"),
                      key=lambda j: j["created"])
        for job in done[:len(jobs) - MAX_JOBS]:
            jobs.pop(job["id"], None)


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def _read_gallery() -> list[dict]:
    """Caller holds gallery_lock."""
    if not GALLERY_PATH.exists():
        return []
    try:
        return json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_gallery(items: list[dict]) -> None:
    """Caller holds gallery_lock. Written to one side and renamed over, so an
    interrupted write cannot leave a half-file — a truncated gallery.json reads
    back as empty, which would lose the whole library."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = GALLERY_PATH.with_name(GALLERY_PATH.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, GALLERY_PATH)


def read_gallery() -> list[dict]:
    with gallery_lock:
        return _read_gallery()


def write_gallery(items: list[dict]) -> None:
    with gallery_lock:
        _write_gallery(items)


def add_images(items: list[dict]) -> None:
    # Read and write under one lock. Batches finish at the same moment, and
    # releasing between the two lets one job overwrite another's images.
    with gallery_lock:
        _write_gallery(items + _read_gallery())


def title_from(p: dict) -> str:
    text = (p.get("description") or "").strip()
    if text:
        return " ".join(text.split()[:8]).strip(" ,.!?-")
    return "Untitled image"


# --------------------------------------------------------------------------- #
# step progress over the ComfyUI websocket (optional dependency)
# --------------------------------------------------------------------------- #
def ws_listener() -> None:
    try:
        import websocket  # websocket-client
    except ImportError:
        return
    while True:
        try:
            url = cfg["comfy_url"].replace("http://", "ws://").replace(
                "https://", "wss://")
            ws = websocket.WebSocket()
            ws.connect(f"{url}/ws?clientId={client.client_id}", timeout=10)
            current = None
            while True:
                raw = ws.recv()
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data") or {}
                pid = data.get("prompt_id") or current
                if mtype == "execution_start":
                    current = data.get("prompt_id")
                elif mtype == "progress" and pid:
                    ws_progress.setdefault(pid, {}).update(
                        value=data.get("value", 0), max=data.get("max", 0))
                elif mtype in ("execution_success", "execution_error") and pid:
                    ws_progress.pop(pid, None)
        except Exception:
            time.sleep(4)


# --------------------------------------------------------------------------- #
# generation job
# --------------------------------------------------------------------------- #
def run_job(job_id: str, params: dict) -> None:
    prompt_id = ""

    def set_state(**kw):
        with jobs_lock:
            if job_id in jobs:      # it may have aged out of the list
                jobs[job_id].update(kw)

    try:
        set_state(stage="Building the graph", pct=2, vague=True)
        built = client.build(params)
        prompt_id = client.queue(built["prompt"])
        set_state(prompt_id=prompt_id, seed=built["seed"], pct=5, vague=True,
                  stage="Queued in ComfyUI")

        started = time.time()
        while True:
            time.sleep(1.0)
            # Read the flag under the lock, then let it go: set_state() takes
            # the same lock, and jobs_lock is not reentrant.
            with jobs_lock:
                cancelled = bool(jobs[job_id].get("cancelled"))
            if cancelled:
                client.cancel(prompt_id)
                set_state(status="cancelled", stage="Cancelled")
                return
            err = client.failed(prompt_id)
            if err:
                set_state(status="error", error=err, stage="Failed")
                return
            outs = client.outputs(prompt_id)
            if outs:
                break
            wp = ws_progress.get(prompt_id) or {}
            value, maximum = wp.get("value", 0), wp.get("max", 0)
            if maximum:
                set_state(pct=round(6 + min(value / maximum, 1) * 88, 1),
                          vague=False, stage=f"Step {value} of {maximum}")
            else:
                # No step progress yet means ComfyUI is loading weights. There
                # is no percentage for that, and the weights are bigger than
                # most cards, so say how long it has been instead of pinning a
                # made-up number on the bar.
                waited = int(time.time() - started)
                been = (f"{waited // 60}m {waited % 60:02d}s" if waited >= 60
                        else f"{waited}s")
                note = (" · the first run is the slow one" if waited > 90
                        else "")
                set_state(pct=min(5 + waited / 4, 12), vague=True,
                          stage=f"Loading the model — {been}{note}")
            if time.time() - started > 3600:
                set_state(status="error", stage="Timed out",
                          error="No image after an hour. Check the ComfyUI log.")
                return

        set_state(stage="Saving", pct=96, vague=False)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for index, item in enumerate(outs):
            image_id = uuid.uuid4().hex[:12]
            ext = Path(item["filename"]).suffix or ".png"
            dest = IMAGES_DIR / f"{image_id}{ext}"
            with client.view(item) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1024 * 256):
                        fh.write(chunk)
            saved.append({
                "id": image_id, "file": dest.name,
                "title": params.get("title") or title_from(params),
                "description": params.get("description", ""),
                "background": params.get("background", ""),
                "style": params.get("style", ""),
                "style_photo": params.get("style_photo", ""),
                "aesthetics": params.get("aesthetics", ""),
                "lighting": params.get("lighting", ""),
                "medium": params.get("medium", ""),
                "regions": params.get("regions") or [],
                "ref_image": params.get("ref_image") or "",
                "ref_mask": params.get("ref_mask") or "",
                "ref_denoise": built.get("ref_denoise"),
                "width": params.get("width"), "height": params.get("height"),
                "steps": params.get("steps"), "cfg": params.get("cfg"),
                "guider_cfg": params.get("guider_cfg"),
                "shift": params.get("shift"),
                "sampler": params.get("sampler"),
                "scheduler": params.get("scheduler"),
                "seed": built["seed"], "batch_index": index,
                "model": built["files"]["cond"],
                "builder": built["builder"],
                "loras": built.get("loras") or [],
                "created": time.time(),
            })
        add_images(saved)
        set_state(status="done", pct=100, stage="Ready", images=saved)
    except ComfyError as exc:
        set_state(status="error", error=str(exc), stage="Failed")
    except Exception as exc:  # noqa: BLE001
        set_state(status="error", error=f"{type(exc).__name__}: {exc}",
                  stage="Failed")
    finally:
        # The websocket drops its entry on execution_success, but a run that
        # errors or is interrupted may never send one.
        if prompt_id:
            ws_progress.pop(prompt_id, None)


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


# --------------------------------------------------------------------------- #
# status / setup
# --------------------------------------------------------------------------- #
def engine_is_stale() -> bool:
    """True when every weight is on disk and the engine's own diffusion-model
    list still does not have them.

    ComfyUI scans its model folders once, at startup. A download that lands
    afterwards is on disk and invisible, so the dropdowns are empty and every
    run dies on "no model" — advice to download what is already there. Only a
    restart makes it look again.
    """
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if not models_dir or not models_dir.is_dir():
        return False
    if bootstrap.missing_models(models_dir, cfg):
        return False        # genuinely missing is a different problem
    try:
        unets = client.unets()
    except Exception:       # noqa: BLE001  nothing to judge if it will not say
        return False
    return not any(MODEL_MARKER in u.lower() for u in unets)


@app.get("/api/status")
def api_status():
    online = comfy_online(cfg["comfy_url"])
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    missing = []
    if models_dir and models_dir.is_dir():
        missing = [m["name"] for m in bootstrap.missing_models(models_dir, cfg)]
    payload = {
        "comfy_online": online,
        # Alive but not answering yet is "starting", not "offline" — the first
        # start loads PyTorch and the models, which takes minutes.
        "comfy_running_managed": comfy_proc.alive(),
        "setup_complete": bool(cfg.get("setup_complete")),
        "missing_models": missing,
        "detected": detect_comfy_dirs(),
        "precisions": {k: {"label": v["label"], "note": v["note"]}
                       for k, v in bootstrap.PRECISIONS.items()},
        "config": {k: cfg.get(k) for k in
                   ("comfy_url", "comfy_dir", "models_dir", "managed",
                    "auto_start_comfy", "torch_index", "precision",
                    "want_manager")},
        "nodes_ready": False, "ready": False,
    }
    if online:
        try:
            payload["capabilities"] = client.capabilities()
            payload["nodes_ready"] = client.has(CORE_NODE)
            payload["samplers"] = client.samplers()
            payload["schedulers"] = client.schedulers()
            payload["unets"] = client.unets()
            payload["clips"] = client.clips()
            payload["vaes"] = client.vaes()
        except Exception as exc:  # noqa: BLE001
            payload["schema_error"] = str(exc)
        # The two silent "nothing works" states, named: an engine that scanned
        # before the weights landed, and an address answered by a different
        # install than the one this app set up.
        payload["stale_models"] = engine_is_stale()
        stats = bootstrap.comfy_stats(cfg["comfy_url"]) or {}
        argv = (stats.get("argv") or [""])[0]
        payload["engine_argv"] = argv
        want = (str(Path(cfg["comfy_dir"])).replace("\\", "/").lower()
                if cfg.get("comfy_dir") else "")
        payload["engine_mismatch"] = bool(
            argv and want and want not in argv.replace("\\", "/").lower())
        payload["engine_managed"] = comfy_proc.alive()
    payload["ready"] = bool(online and payload["nodes_ready"] and not missing)
    return jsonify(payload)


@app.post("/api/setup/start")
def api_setup_start():
    if progress.running:
        return jsonify({"error": "Setup is already running."}), 409
    b = request.get_json(silent=True) or {}
    for key in ("comfy_url", "models_dir", "precision", "want_manager"):
        if key in b:
            cfg[key] = b[key]
    client.url = cfg["comfy_url"].rstrip("/")
    save_config(cfg)
    progress.__init__()
    threading.Thread(target=bootstrap.run_setup,
                     args=(cfg, progress, comfy_proc, b.get("comfy_dir", ""),
                           b.get("mode", "auto")), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/setup/state")
def api_setup_state():
    snap = progress.snapshot(int(request.args.get("since", 0)))
    snap["comfy_tail"] = comfy_proc.tail(12)
    return jsonify(snap)


def _note(msg: str) -> None:
    """Engine actions belong in the engine console, next to its own output."""
    comfy_proc.note(msg)
    progress.log(msg)


def take_over_port(url: str, port: int):
    """Close whatever ComfyUI answers on the port.

    Returns ("manager-reboot", None) when ComfyUI-Manager rebooted it in
    place, ("freed", None) when the port is now empty, or (None, advice)
    when it cannot be done — with advice that names the actual obstacle,
    because "close it yourself" against a windowless process is a treasure
    hunt through Task Manager.
    """
    _note("This ComfyUI was not started here — taking it over.")
    try:
        r = requests.post(f"{url}/manager/reboot", json={}, timeout=5)
        accepted = r.status_code in (200, 201, 204)
    except requests.exceptions.RequestException:
        accepted = True          # the connection dropping is the reboot
    if accepted:
        deadline = time.time() + 10
        while time.time() < deadline:
            if not comfy_online(url):
                _note("ComfyUI-Manager took the reboot; waiting for the "
                      "engine to come back.")
                return "manager-reboot", None
            time.sleep(0.5)
        _note("ComfyUI-Manager did not take the reboot; stopping the "
              "process instead.")

    def settled_free() -> bool:
        # a supervisor (ComfyUI Desktop, a launcher .bat) respawns in under
        # a second — quiet is only free once it stays quiet
        time.sleep(2.0)
        return not comfy_online(url) and not bootstrap.port_pids(port)

    first_pids: list[int] = []
    denied = False
    for attempt in range(3):
        pids = bootstrap.port_pids(port)
        if attempt == 0:
            first_pids = pids
        if not pids:
            if not comfy_online(url) and settled_free():
                return "freed", None
            if not comfy_online(url):
                _note("It came straight back — something restarted it.")
                continue
            return None, (f"Something answers on port {port} but its process "
                          "could not be found — it may belong to another "
                          "user account. Close it in Task Manager, then "
                          "press Start ComfyUI.")
        for pid in pids:
            cmd = bootstrap.pid_cmdline(pid)
            _note(f"Port {port} is held by pid {pid}"
                  + (f": {cmd[:120]}" if cmd else " (command line unreadable)"))
            if cmd and not any(k in cmd.lower()
                               for k in ("python", "main.py", "comfy")):
                return None, (f"Port {port} is held by something that does "
                              f"not look like ComfyUI ({cmd[:90]}). Close it "
                              "yourself, or point Settings at a different "
                              "address.")
        for pid in pids:
            said = bootstrap.kill_pid(pid)
            _note(f"Stopping pid {pid} — {said or 'no reply'}")
            if "denied" in (said or "").lower() \
                    or "access" in (said or "").lower():
                denied = True
        deadline = time.time() + 8
        while comfy_online(url) and time.time() < deadline:
            time.sleep(0.5)
        if not comfy_online(url):
            if settled_free():
                return "freed", None
            _note("It came straight back — something restarted it.")
            continue
        _note("Still answering — trying again.")

    now = bootstrap.port_pids(port)
    if denied:
        return None, ("The system refused to stop it (access denied) — it was "
                      "started as another user, or as administrator. Run "
                      "Ideogram Studio as administrator once, or close it in "
                      "Task Manager, then press Start ComfyUI.")
    if now and set(now) != set(first_pids):
        return None, ("It keeps coming back under a new process id — "
                      "something is supervising it (ComfyUI Desktop, or a "
                      "launcher script). Close that application, then press "
                      "Start ComfyUI.")
    return None, ("It would not close. The Engine console shows what was "
                  "tried; close it in Task Manager, then press Start "
                  "ComfyUI.")


def _refresh_schema_when_up() -> None:
    """After a (re)start, drop the cached schema the moment the engine
    answers — otherwise the fresh model scan hides behind the old cache
    for up to two minutes, and the page still says the weights are missing."""
    def wait():
        if bootstrap.wait_for_comfy(cfg["comfy_url"], timeout=900):
            try:
                client.schema(force=True)
            except Exception:
                pass
    threading.Thread(target=wait, daemon=True).start()


@app.post("/api/comfy/start")
def api_comfy_start():
    if comfy_online(cfg["comfy_url"]):
        return jsonify({"ok": True, "already": True})
    if comfy_proc.alive():
        return jsonify({"ok": True, "starting": True})
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        if cfg.get("comfy_dir") and not cfg.get("managed", True):
            return jsonify({"error": "Ideogram Studio does not know which "
                            "Python that ComfyUI runs on, so it will not start "
                            "it. Start ComfyUI yourself, then press "
                            "Recheck."}), 400
        return jsonify({"error": "Run setup first."}), 400
    _note("Starting ComfyUI…")
    comfy_proc.start(py, Path(cfg["comfy_dir"]),
                     int(cfg["comfy_url"].rsplit(":", 1)[-1]), progress)
    _refresh_schema_when_up()
    return jsonify({"ok": True})


@app.post("/api/comfy/restart")
def api_comfy_restart():
    """Stop and start ComfyUI, so it rescans its model folders and loads
    newly installed nodes — the two things only a restart does.

    An engine this app did not start (an orphan from an earlier run, or one
    launched by hand) is taken over rather than declared unreachable: first
    ComfyUI-Manager's own reboot, and failing that the process holding the
    configured port is verified to look like ComfyUI and stopped, then a
    managed one starts in its place. The old advice — "close it yourself" —
    asked people to hunt a windowless python in Task Manager.
    """
    url = cfg["comfy_url"]
    port = int(url.rsplit(":", 1)[-1])
    py = bootstrap.comfy_python(cfg)
    can_start = bool(cfg.get("comfy_dir") and py)

    if comfy_proc.alive():
        if not can_start:
            return jsonify({"error": "Run setup first."}), 400
        _note("Restarting the managed engine…")
        comfy_proc.stop()
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress)
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "managed"})

    if not comfy_online(url):
        if not can_start:
            return jsonify({"error": "Run setup first."}), 400
        _note("Starting ComfyUI…")
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress)
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "started"})

    # online, but not ours — take it over
    how, advice = take_over_port(url, port)
    if advice:
        return jsonify({"error": advice}), 409
    if how == "manager-reboot":
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "manager-reboot"})
    if not can_start:
        return jsonify({"ok": True, "how": "stopped",
                        "note": "Stopped it. This app has no ComfyUI of its "
                                "own to start — run setup, or start yours "
                                "again yourself."})
    _note("Starting a managed engine in its place…")
    comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress)
    _refresh_schema_when_up()
    return jsonify({"ok": True, "how": "takeover"})


@app.get("/api/comfy/log")
def api_comfy_log():
    """The engine's own console — the visible cue that it is starting,
    started, or telling you exactly what failed to import."""
    n = min(max(int(request.args.get("n", 80)), 1), 400)
    return jsonify({"lines": comfy_proc.tail(n),
                    "running": comfy_proc.alive(),
                    "online": comfy_online(cfg["comfy_url"])})


@app.post("/api/config")
def api_config():
    b = request.get_json(silent=True) or {}
    for key in ("comfy_url", "comfy_dir", "models_dir", "auto_start_comfy",
                "torch_index", "precision", "want_manager"):
        if key in b:
            cfg[key] = b[key]
    client.url = cfg["comfy_url"].rstrip("/")
    save_config(cfg)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# dependencies / tasks
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    live = client if comfy_online(cfg["comfy_url"]) else None
    return jsonify({"items": manager.dependencies(cfg, live,
                                                  starting=comfy_proc.alive()),
                    "torch_index": cfg.get("torch_index", "")})


@app.post("/api/deps/<path:dep_id>/install")
def api_dep_install(dep_id: str):
    b = request.get_json(silent=True) or {}
    if b.get("torch_index") is not None:
        cfg["torch_index"] = b["torch_index"]
        save_config(cfg)
    try:
        return jsonify({"ok": True,
                        "task": manager.install_dependency(dep_id, cfg, b).view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    task_id = request.args.get("id", "")
    since = int(request.args.get("since", 0))
    if task_id:
        task = manager.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"]) for t in manager.TASKS.list()[:25]])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = manager.TASKS.get(task_id)
    if task:
        task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# huggingface
# --------------------------------------------------------------------------- #
@app.get("/api/hf/settings")
def api_hf_settings():
    token = cfg.get("hf_token") or ""
    return jsonify({"endpoint": cfg.get("hf_endpoint") or manager.DEFAULT_ENDPOINT,
                    "token_set": bool(token),
                    "token_hint": ("…" + token[-4:]) if len(token) > 4 else "",
                    "repo": cfg.get("hf_repo") or bootstrap.MODEL_REPO,
                    "curated": manager.curated(cfg),
                    "folders": manager.MODEL_FOLDERS,
                    "models_dir": cfg.get("models_dir", "")})


@app.post("/api/hf/settings")
def api_hf_settings_save():
    b = request.get_json(silent=True) or {}
    if "token" in b:
        cfg["hf_token"] = (b["token"] or "").strip()
    if b.get("endpoint") is not None:
        cfg["hf_endpoint"] = b["endpoint"].strip() or manager.DEFAULT_ENDPOINT
    if b.get("repo"):
        cfg["hf_repo"] = b["repo"].strip()
    if b.get("models_dir"):
        cfg["models_dir"] = b["models_dir"].strip()
    if b.get("precision"):
        cfg["precision"] = b["precision"]
    save_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/hf/browse")
def api_hf_browse():
    repo = (request.args.get("repo") or cfg.get("hf_repo") or "").strip()
    try:
        data = manager.hf_browse(cfg, repo)
        cfg["hf_repo"] = repo
        save_config(cfg)
        return jsonify(data)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.post("/api/hf/download")
def api_hf_download():
    b = request.get_json(silent=True) or {}
    try:
        if b.get("set"):
            if b.get("precision"):
                cfg["precision"] = b["precision"]
                save_config(cfg)
            tasks = manager.download_set(cfg)
            if not tasks:
                return jsonify({"ok": True, "tasks": [],
                                "note": "Everything in that set is already here."})
            return jsonify({"ok": True, "tasks": [t.view() for t in tasks]})
        path = (b.get("path") or "").strip()
        if not path:
            return jsonify({"error": "Pick a file to download."}), 400
        task = manager.hf_download(cfg, b.get("repo") or cfg.get("hf_repo")
                                   or bootstrap.MODEL_REPO, path,
                                   b.get("folder") or "")
        return jsonify({"ok": True, "task": task.view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/hf/local")
def api_hf_local():
    return jsonify({"models": manager.local_models(cfg),
                    "models_dir": cfg.get("models_dir", "")})


@app.delete("/api/hf/local")
def api_hf_delete():
    b = request.get_json(silent=True) or {}
    try:
        manager.delete_model(cfg, b.get("folder", ""), b.get("name", ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
def api_generate():
    params = request.get_json(silent=True) or {}
    has_text = bool((params.get("description") or "").strip())
    has_region = any((r.get("desc") or r.get("text"))
                     for r in (params.get("regions") or []))
    if not has_text and not has_region:
        return jsonify({"error": "Describe the image, or add a region with a "
                                 "description."}), 400
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running. Start it from the "
                                 "Engine page."}), 503
    # Without this the run fails deep in the graph builder with "no files in
    # diffusion_models" — advice to download weights that are already on disk.
    if engine_is_stale():
        return jsonify({"error": "The weights are on disk, but this ComfyUI "
                                 "started before they landed and has not "
                                 "rescanned — so it cannot see them. Press "
                                 "Restart ComfyUI on the Engine page."}), 409
    runs = max(1, min(int(params.get("runs") or 1), 4))
    created = []
    for _ in range(runs):
        job_id = uuid.uuid4().hex[:12]
        with jobs_lock:
            prune_jobs()
            jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
                            "vague": True,
                            "stage": "Starting", "created": time.time(),
                            "title": params.get("title") or title_from(params)}
        threading.Thread(target=run_job, args=(job_id, dict(params)),
                         daemon=True).start()
        created.append(job_id)
        time.sleep(0.2)
    return jsonify({"jobs": created})


@app.get("/api/jobs")
def api_jobs():
    with jobs_lock:
        prune_jobs()
        active = [j for j in jobs.values()
                  if j["status"] == "running" or time.time() - j["created"] < 180]
        return jsonify(sorted(active, key=lambda j: j["created"], reverse=True))


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    # Just raise the flag. The job thread owns its prompt_id, so only it can
    # cancel the right prompt — interrupting from here would stop whichever
    # job the engine happens to be running.
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["cancelled"] = True
    return jsonify({"ok": True})


@app.get("/api/ref-preview")
def api_ref_preview():
    """Thumbnail for a reference already sitting in ComfyUI's input folder —
    the page cannot read that folder itself, so relay ComfyUI's own /view."""
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "No name."}), 400
    sub, _, fname = name.rpartition("/")
    try:
        r = client.view({"filename": fname, "subfolder": sub, "type": "input"})
        r.raise_for_status()
        return Response(r.iter_content(1024 * 64),
                        content_type=r.headers.get("Content-Type", "image/png"))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"ComfyUI could not serve it: {exc}"}), 502


@app.post("/api/image/<image_id>/as-reference")
def api_image_as_reference(image_id: str):
    """Push a gallery image into ComfyUI's input folder so it can seed a run."""
    for item in read_gallery():
        if item["id"] == image_id:
            path = IMAGES_DIR / item["file"]
            if not path.exists():
                return jsonify({"error": "That file is missing."}), 404

            class _Upload:                     # the shape upload_image reads
                filename = path.name
                stream = open(path, "rb")
                mimetype = mimetypes.guess_type(path.name)[0] or "image/png"
            try:
                return jsonify({"ok": True,
                                "name": client.upload_image(_Upload())})
            except Exception as exc:  # noqa: BLE001
                return jsonify({"error": str(exc)}), 502
            finally:
                _Upload.stream.close()
    return jsonify({"error": "Image not found."}), 404


@app.post("/api/upload-image")
def api_upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file received."}), 400
    try:
        return jsonify({"ok": True,
                        "name": client.upload_image(request.files["file"])})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
@app.get("/api/images")
def api_images():
    return jsonify(read_gallery())


@app.get("/api/image/<image_id>")
def api_image(image_id: str):
    for item in read_gallery():
        if item["id"] == image_id:
            path = IMAGES_DIR / item["file"]
            if not path.exists():
                return jsonify({"error": "That file is missing."}), 404
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            return send_file(path, mimetype=mime, conditional=True,
                             download_name=f"{item['title']}{path.suffix}")
    return jsonify({"error": "Image not found."}), 404


@app.delete("/api/image/<image_id>")
def api_image_delete(image_id: str):
    with gallery_lock:
        keep, drop = [], []
        for item in _read_gallery():
            (drop if item["id"] == image_id else keep).append(item)
        if drop:
            _write_gallery(keep)
    for item in drop:  # unlink outside the lock, once the entry is really gone
        try:
            (IMAGES_DIR / item["file"]).unlink(missing_ok=True)
        except OSError:
            pass
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# loras
#
# There is no in-app LoRA browser: the Models page can already pull any file
# from any HuggingFace repo into any model folder, loras included. These routes
# just report what is on disk so the picker can list it.
# --------------------------------------------------------------------------- #
@app.get("/api/loras")
def api_loras():
    live, supported = [], False
    if comfy_online(cfg["comfy_url"]):
        try:
            live = client.loras()
            supported = client.has("LoraLoaderModelOnly") or client.has("LoraLoader")
        except Exception:  # noqa: BLE001
            live, supported = [], False
    return jsonify({"installed": manager.loras_installed(cfg),
                    "known_to_comfy": live, "supported": supported})


@app.delete("/api/loras")
def api_lora_delete():
    b = request.get_json(silent=True) or {}
    try:
        manager.delete_lora(cfg, b.get("name", ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


def ensure_engine_at_boot() -> None:
    """A launch ends with a working engine, without a button pressed.

    Offline: start the managed one, as the old inline auto-start did. Online
    and healthy: adopt it and say so. Online but useless — a stale scan hiding
    the weights, installed nodes it never loaded, or a different install
    squatting the port — replace it, with the same looks-like-ComfyUI guard
    the Restart button uses. An external-mode setup (managed False) is never
    touched: that engine is the person's own.
    """
    if not (cfg.get("setup_complete") and cfg.get("auto_start_comfy", True)):
        return
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return
    url = cfg["comfy_url"]
    port = int(url.rsplit(":", 1)[-1])

    if not comfy_online(url):
        _note("Restarting ComfyUI from the last setup…")
        comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress)
        _refresh_schema_when_up()
        return

    # something already answers — decide between adopting and replacing
    reasons = []
    try:
        client.schema(force=True)
        has_nodes = client.has(CORE_NODE)
        unets = client.unets()
    except Exception as exc:  # noqa: BLE001
        _note(f"The engine already running would not describe itself "
              f"({exc}) — leaving it alone.")
        return
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    weights_here = bool(models_dir and models_dir.is_dir() and
                        not bootstrap.missing_models(models_dir, cfg))
    if weights_here and not any(MODEL_MARKER in u.lower() for u in unets):
        reasons.append("it started before the weights landed")
    if not has_nodes and weights_here:
        reasons.append("the Ideogram 4 nodes are not loaded")
    for node in bootstrap.CUSTOM_NODES:
        marker = NODE_MARKERS.get(node["id"])
        if marker and bootstrap.node_installed(Path(cfg["comfy_dir"]), node) \
                and not client.has(marker):
            reasons.append(f"{node['label']} is installed but not loaded")
    stats = bootstrap.comfy_stats(url) or {}
    argv = (stats.get("argv") or [""])[0]
    want = str(Path(cfg["comfy_dir"])).replace("\\", "/").lower()
    if argv and want and want not in argv.replace("\\", "/").lower():
        reasons.append("a different install is answering the address")

    if not reasons:
        _note(f"Adopting the ComfyUI already running at {url}.")
        return
    if not cfg.get("managed", True):
        _note("The engine already running has problems ("
              + "; ".join(reasons) + ") but it is yours, not this app's — "
              "restart it yourself, or press Restart ComfyUI.")
        return
    _note("The engine already running is no use as it stands — "
          + "; ".join(reasons) + ". Replacing it.")
    how, advice = take_over_port(url, port)
    if advice:
        _note(advice)
        return
    if how == "manager-reboot":
        _refresh_schema_when_up()
        return
    _note("Starting a managed engine in its place…")
    comfy_proc.start(py, Path(cfg["comfy_dir"]), port, progress)
    _refresh_schema_when_up()


# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=ws_listener, daemon=True).start()
    # the engine comes up on its own; the page can open meanwhile
    threading.Thread(target=ensure_engine_at_boot, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  Ideogram Studio  →  {url}\n")
    if os.environ.get("IDEOGRAM_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        comfy_proc.stop()


if __name__ == "__main__":
    main()
