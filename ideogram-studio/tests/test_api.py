"""The engine kit over HTTP: a real server.py against real engine processes.

Nothing here is mocked at the process level. The app starts, adopts, stops and
replaces actual children, and the diagnoses it gives ("something is supervising
it", "that is not ComfyUI") are proved by arranging exactly that situation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (Suite, Workspace, comfy, engine_log,  # noqa: E402
                     fake_install, fake_weights, foreign_listener, free_port,
                     status, studio, supervised_comfy, wait_for)


def run(slow: bool = False) -> Suite:
    s = Suite("api")

    # -- the console endpoint ------------------------------------------------
    with comfy() as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models) as app:
            page = requests.get(app.url + "/", timeout=10)
            s.check("the page is served, with the engine console on it",
                    page.ok and 'id="engine-log"' in page.text
                    and 'id="engine-state"' in page.text
                    and 'id="btnRestartEngine"' in page.text)
            log = engine_log(app.url, 5)
            s.check("the log endpoint reports running and online separately",
                    log["online"] is True and log["running"] is False
                    and isinstance(log["lines"], list), str(log)[:90])
            s.equal("n is clamped to at most 400",
                    requests.get(app.url + "/api/comfy/log?n=9000",
                                 timeout=10).status_code, 200)
            # a restart against an engine this app cannot replace still says
            # what it did, and the console keeps the record of it
            requests.post(app.url + "/api/comfy/restart", timeout=60)
            lines = "\n".join(engine_log(app.url)["lines"])
            s.check("the app's own actions are noted in the engine console",
                    "[Ideogram Studio]" in lines
                    and "taking it over" in lines, lines[-120:])
            s.check("restart with no install of our own stops it and says so",
                    wait_for(lambda: not engine_log(app.url)["online"], 20))

    # -- weights on disk, engine started before they landed -------------------
    # ComfyUI scans its model folders once, at startup; this is the silent
    # "nothing works" a real install hits: every dropdown empty, and generate
    # failing with a download hint for a file that is already there.
    with comfy(MOCK_BLANK_UNETS="1") as stale, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(stale.url, ws / "data", models) as app:
            st = status(app.url)
            s.check("status names the stale-scan state",
                    st["stale_models"] is True and st["comfy_online"]
                    and st["missing_models"] == [], str(st)[:120])
            r = requests.post(app.url + "/api/generate",
                              json={"description": "doomed by a stale scan"},
                              timeout=30)
            s.check("generate refuses with the fix, not a download hint",
                    r.status_code == 409
                    and "Restart" in r.json()["error"]
                    and "on disk" in r.json()["error"],
                    str(r.json())[:110])
            r = requests.post(app.url + "/api/comfy/restart", timeout=60)
            s.check("restart takes over an engine it does not own",
                    r.ok and r.json()["how"] == "stopped"
                    and "run setup" in r.json()["note"].lower(),
                    str(r.json())[:90])
            s.check("the foreign engine is actually gone",
                    wait_for(lambda: not engine_log(app.url)["online"], 20))

    # -- nothing running at all: restart is a start ---------------------------
    with Workspace() as ws:
        install = fake_install(ws)
        quiet = f"http://127.0.0.1:{free_port()}"
        with studio(quiet, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable) as app:
            r = requests.post(app.url + "/api/comfy/restart", timeout=60)
            s.check("restart with nothing there just starts one",
                    r.ok and r.json()["how"] == "started", str(r.json())[:90])
            s.check("and it comes up managed",
                    wait_for(lambda: (lambda x: x["comfy_online"]
                             and x["engine_managed"])(status(app.url)), 45))
            before = status(app.url)
            s.check("the managed engine sees the weights it was started with",
                    before["stale_models"] is False and before["ready"]
                    and any("ideogram4" in u.lower() for u in before["unets"]),
                    str(before.get("unets"))[:90])

            # -- a managed restart: same app, new process -------------------
            was = [ln for ln in engine_log(app.url)["lines"]
                   if "Starting server" in ln]
            r = requests.post(app.url + "/api/comfy/restart", timeout=60)
            s.check("restarting our own engine is the plain managed kind",
                    r.ok and r.json()["how"] == "managed", str(r.json())[:90])
            s.check("it comes back, having started a second time",
                    wait_for(lambda: status(app.url)["comfy_online"]
                             and len([ln for ln in engine_log(app.url)["lines"]
                                      if "Starting server" in ln]) > len(was),
                             45))

    # -- a weight that lands after the engine did -----------------------------
    # The stale flag is the point of the whole kit, so prove both halves of it
    # against one engine: blind first, seeing after the restart.
    with Workspace() as ws:
        install = fake_install(ws, stale_first_boot=True)
        quiet = f"http://127.0.0.1:{free_port()}"
        with studio(quiet, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=True, managed=True) as app:
            s.check("the engine started by boot is stale on its first scan",
                    wait_for(lambda: status(app.url).get("stale_models")
                             is True, 45))
            st = status(app.url)
            s.check("nothing is missing from disk — only from the engine",
                    st["missing_models"] == [] and st["unets"] == [],
                    str(st.get("unets")))
            r = requests.post(app.url + "/api/comfy/restart", timeout=60)
            s.check("restarting it is the managed route", r.ok
                    and r.json()["how"] == "managed", str(r.json())[:90])
            s.check("after the restart the weights are in its list",
                    wait_for(lambda: status(app.url).get("stale_models")
                             is False, 60))
            s.check("and the app calls itself ready",
                    status(app.url)["ready"])

    # -- full takeover: an orphan on the port, replaced by a managed engine ---
    with comfy() as orphan, Workspace() as ws:
        install = fake_install(ws)
        with studio(orphan.url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable) as app:
            st = status(app.url)
            s.check("before: online, but not ours",
                    st["comfy_online"] and not st["engine_managed"])
            r = requests.post(app.url + "/api/comfy/restart", timeout=120)
            s.check("restart closes the orphan and starts a managed engine",
                    r.ok and r.json()["how"] == "takeover", str(r.json())[:90])
            s.check("the orphan process is gone",
                    wait_for(lambda: orphan.proc.poll() is not None, 20))
            s.check("the managed engine comes up in its place",
                    wait_for(lambda: (lambda x: x["comfy_online"]
                             and x["engine_managed"])(status(app.url)), 60))
            s.check("and it sees the weights — no stale scan",
                    status(app.url)["stale_models"] is False)

    # -- a supervised engine: the diagnosis is the whole value ----------------
    with supervised_comfy() as sup, Workspace() as ws:
        install = fake_install(ws)
        with studio(sup.url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable) as app:
            r = requests.post(app.url + "/api/comfy/restart", timeout=240)
            s.check("a respawning engine is diagnosed, not shrugged at",
                    r.status_code == 409
                    and "supervising" in r.json()["error"],
                    str(r.json())[:130])
            lines = "\n".join(engine_log(app.url)["lines"])
            s.check("the console shows what was tried",
                    "Stopping pid" in lines
                    and "came straight back" in lines, lines[-140:])

    # -- something that is not ComfyUI at all ---------------------------------
    listener, label = foreign_listener()
    if listener is None:
        s.skip("a non-ComfyUI process on the port is named, not killed", label)
    else:
        with listener as squatter, Workspace() as ws:
            install = fake_install(ws)
            with studio(squatter.url, ws / "data", install / "models",
                        comfy_dir=str(install),
                        python=sys.executable) as app:
                r = requests.post(app.url + "/api/comfy/restart", timeout=120)
                s.check("a stranger on the port is refused with its name",
                        r.status_code == 409
                        and "does not look like ComfyUI" in r.json()["error"]
                        and label in r.json()["error"],
                        str(r.json())[:150])
                s.check("and it is left running — this app does not kill "
                        "what it cannot identify",
                        squatter.proc.poll() is None)

    # -- a different install answering the address ----------------------------
    with comfy(MOCK_COMFY_ROOT="/opt/somebody-elses/ComfyUI") as mock, \
            Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models,
                    comfy_dir="/opt/mine/ComfyUI") as app:
            st = status(app.url)
            s.check("a foreign engine on the address is called out",
                    st["engine_mismatch"] is True
                    and "somebody-elses" in st["engine_argv"], str(st)[:120])
    with comfy(MOCK_COMFY_ROOT="/opt/mine/ComfyUI") as mock, Workspace() as ws:
        models = ws / "models"
        fake_weights(models)
        with studio(mock.url, ws / "data", models,
                    comfy_dir="/opt/mine/ComfyUI") as app:
            s.check("the right engine on the address is not",
                    status(app.url)["engine_mismatch"] is False)

    # -- boot: a quiet port ---------------------------------------------------
    with Workspace() as ws:
        install = fake_install(ws)
        quiet = f"http://127.0.0.1:{free_port()}"
        with studio(quiet, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=True) as app:
            s.check("launch boots a managed engine with no clicks",
                    wait_for(lambda: (lambda x: x.get("comfy_online")
                             and x.get("engine_managed"))(status(app.url)), 60))
            s.check("and it comes up ready",
                    status(app.url)["ready"])

    # -- boot: a healthy engine is adopted ------------------------------------
    with comfy() as healthy, Workspace() as ws:
        install = fake_install(ws)
        with studio(healthy.url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=True, managed=True) as app:
            s.check("a healthy engine on the port is adopted, and says so",
                    wait_for(lambda: "Adopting" in "\n".join(
                        engine_log(app.url)["lines"]), 40))
            st = status(app.url)
            s.check("it is left alone — online, not ours, still alive",
                    st["comfy_online"] and not st["engine_managed"]
                    and healthy.proc.poll() is None)

    # -- boot: a stale orphan is replaced -------------------------------------
    with comfy(MOCK_BLANK_UNETS="1") as orphan, Workspace() as ws:
        install = fake_install(ws)
        with studio(orphan.url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=True, managed=True) as app:
            s.check("a stale orphan on the port is replaced by itself",
                    wait_for(lambda: (lambda x: x.get("engine_managed")
                             and x.get("comfy_online")
                             and x.get("stale_models") is False)(
                                 status(app.url)), 90))
            joined = "\n".join(engine_log(app.url)["lines"])
            s.check("the console narrates the boot takeover and its reason",
                    "Replacing it" in joined and "Stopping pid" in joined
                    and "before the weights landed" in joined,
                    joined[-160:])

    # -- boot: an external engine is never touched ----------------------------
    with comfy(MOCK_BLANK_UNETS="1") as mine, Workspace() as ws:
        install = fake_install(ws)
        with studio(mine.url, ws / "data", install / "models",
                    comfy_dir=str(install), python=sys.executable,
                    auto_start_comfy=True, managed=False) as app:
            s.check("an engine in external mode is diagnosed, not seized",
                    wait_for(lambda: "it is yours, not this app's"
                             in "\n".join(engine_log(app.url)["lines"]), 40))
            time.sleep(1.0)
            s.check("it is still running, and still not ours",
                    mine.proc.poll() is None
                    and status(app.url)["engine_managed"] is False)
    return s
