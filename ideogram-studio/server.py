"""
server.py - Ideogram Studio backend.

Run:  python server.py        (opens http://127.0.0.1:7802)
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import manager
from bootstrap import (APP_DIR, ComfyProcess, Progress, comfy_online,
                       detect_comfy_dirs, load_config, save_config)
from comfy import ComfyClient, ComfyError

DATA_DIR = APP_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
GALLERY_PATH = DATA_DIR / "gallery.json"
WEB_DIR = APP_DIR / "web"
PORT = int(os.environ.get("IDEOGRAM_STUDIO_PORT", "7802"))

app = Flask(__name__, static_folder=None)

cfg = load_config()
progress = Progress()
comfy_proc = ComfyProcess()
client = ComfyClient(cfg["comfy_url"])

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
gallery_lock = threading.Lock()
ws_progress: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# gallery
# --------------------------------------------------------------------------- #
def read_gallery() -> list[dict]:
    with gallery_lock:
        if not GALLERY_PATH.exists():
            return []
        try:
            return json.loads(GALLERY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []


def write_gallery(items: list[dict]) -> None:
    with gallery_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        GALLERY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def add_images(items: list[dict]) -> None:
    gallery = read_gallery()
    write_gallery(items + gallery)


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
    def set_state(**kw):
        with jobs_lock:
            jobs[job_id].update(kw)

    try:
        set_state(stage="Building the graph", pct=2)
        built = client.build(params)
        prompt_id = client.queue(built["prompt"])
        set_state(prompt_id=prompt_id, seed=built["seed"], pct=5,
                  stage="Queued in ComfyUI")

        started = time.time()
        while True:
            time.sleep(1.0)
            with jobs_lock:
                if jobs[job_id].get("cancelled"):
                    client.interrupt()
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
                          stage=f"Step {value} of {maximum}")
            else:
                set_state(pct=min(5 + (time.time() - started) / 4, 12),
                          stage="Loading the model")
            if time.time() - started > 3600:
                set_state(status="error", stage="Timed out",
                          error="No image after an hour. Check the ComfyUI log.")
                return

        set_state(stage="Saving", pct=96)
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
@app.get("/api/status")
def api_status():
    online = comfy_online(cfg["comfy_url"])
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    missing = []
    if models_dir and models_dir.is_dir():
        missing = [m["name"] for m in bootstrap.missing_models(models_dir, cfg)]
    payload = {
        "comfy_online": online,
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
            payload["nodes_ready"] = client.has("DualModelGuider")
            payload["samplers"] = client.samplers()
            payload["schedulers"] = client.schedulers()
            payload["unets"] = client.unets()
            payload["clips"] = client.clips()
            payload["vaes"] = client.vaes()
        except Exception as exc:  # noqa: BLE001
            payload["schema_error"] = str(exc)
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


@app.post("/api/comfy/start")
def api_comfy_start():
    if comfy_online(cfg["comfy_url"]):
        return jsonify({"ok": True, "already": True})
    py = bootstrap.comfy_python(cfg)
    if not cfg.get("comfy_dir") or not py:
        return jsonify({"error": "Run setup first."}), 400
    comfy_proc.start(py, Path(cfg["comfy_dir"]),
                     int(cfg["comfy_url"].rsplit(":", 1)[-1]), progress)
    return jsonify({"ok": True})


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
    return jsonify({"items": manager.dependencies(cfg, live),
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
    runs = max(1, min(int(params.get("runs") or 1), 4))
    created = []
    for _ in range(runs):
        job_id = uuid.uuid4().hex[:12]
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
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
        active = [j for j in jobs.values()
                  if j["status"] == "running" or time.time() - j["created"] < 180]
        return jsonify(sorted(active, key=lambda j: j["created"], reverse=True))


@app.post("/api/jobs/<job_id>/cancel")
def api_job_cancel(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["cancelled"] = True
    client.interrupt()
    return jsonify({"ok": True})


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
    items = read_gallery()
    for item in items:
        if item["id"] == image_id:
            try:
                (IMAGES_DIR / item["file"]).unlink(missing_ok=True)
            except OSError:
                pass
    write_gallery([i for i in items if i["id"] != image_id])
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


# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=ws_listener, daemon=True).start()
    if cfg.get("setup_complete") and cfg.get("auto_start_comfy", True) \
            and cfg.get("comfy_dir") and bootstrap.comfy_python(cfg) \
            and not comfy_online(cfg["comfy_url"]):
        progress.log("Restarting ComfyUI from the last setup…")
        comfy_proc.start(bootstrap.comfy_python(cfg), Path(cfg["comfy_dir"]),
                         int(cfg["comfy_url"].rsplit(":", 1)[-1]), progress)
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
