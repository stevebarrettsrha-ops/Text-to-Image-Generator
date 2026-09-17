"""
comfy.py - talks to ComfyUI and builds the Ideogram 4 graph.

Built from /object_info rather than a stored workflow: node inputs get renamed
between releases, and reading the schema turns a rename into a clear message
instead of a silently wrong value.

The graph mirrors ideogram4.json, minus the canvas helper nodes (rgthree seed,
resolution picker, preview nodes) whose jobs the front end does itself:

  UNETLoader(cond) ─ ModelSamplingAuraFlow ─ CFGOverride ─┐
                                                          ├─ DualModelGuider ─┐
  UNETLoader(uncond) ───────────────────────── model_negative ┘               │
  CLIPLoader ─ CLIPTextEncode ─┬─ positive ───────────────────┘               │
                               └─ ConditioningZeroOut ─ negative              │
  Ideogram4PromptBuilderKJ ─ prompt ─ CLIPTextEncode                          │
                                                                              ▼
  RandomNoise + KSamplerSelect + BasicScheduler + EmptyFlux2LatentImage ─ SamplerCustomAdvanced
                                                                              │
                                                    VAELoader ─ VAEDecode ─ SaveImage
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid

import requests

PROMPT_BUILDER = "Ideogram4PromptBuilderKJ"
# From the DeverStyle model card, not a guess.
LORA_DEFAULT_STRENGTH = 0.6


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188") -> None:
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None
        self._schema_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def schema(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._schema is None or time.time() - self._schema_at > 120:
                r = requests.get(f"{self.url}/object_info", timeout=30)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def has(self, class_type: str) -> bool:
        return class_type in self.schema()

    def node_inputs(self, class_type: str) -> dict:
        info = self.schema().get(class_type)
        if not info:
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. Ideogram 4 needs a "
                "recent ComfyUI plus the KJNodes pack — check the Engine page.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    REQUIRED_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode",
                      "ConditioningZeroOut", "DualModelGuider", "RandomNoise",
                      "KSamplerSelect", "BasicScheduler", "SamplerCustomAdvanced",
                      "VAEDecode", "SaveImage")

    def ensure_supported(self) -> None:
        missing = [n for n in self.REQUIRED_NODES if not self.has(n)]
        if missing:
            raise ComfyError(
                "This ComfyUI cannot run Ideogram 4 — it is missing "
                + ", ".join(missing) + ". Update ComfyUI from the Engine page, "
                "then restart it.")

    def _enum(self, class_type: str, name: str) -> list[str]:
        try:
            spec = self.node_inputs(class_type).get(name)
        except ComfyError:
            return []
        if spec and isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
        return []

    def unets(self) -> list[str]:
        return self._enum("UNETLoader", "unet_name")

    def clips(self) -> list[str]:
        return self._enum("CLIPLoader", "clip_name")

    def vaes(self) -> list[str]:
        return self._enum("VAELoader", "vae_name")

    def loras(self) -> list[str]:
        for cls in ("LoraLoaderModelOnly", "LoraLoader"):
            if self.has(cls):
                vals = self._enum(cls, "lora_name")
                if vals:
                    return vals
        return []

    def samplers(self) -> list[str]:
        return self._enum("KSamplerSelect", "sampler_name")

    def schedulers(self) -> list[str]:
        return self._enum("BasicScheduler", "scheduler")

    def clip_types(self) -> list[str]:
        return self._enum("CLIPLoader", "type")

    def capabilities(self) -> dict:
        return {"prompt_builder": self.has(PROMPT_BUILDER),
                "lora": self.has("LoraLoaderModelOnly") or self.has("LoraLoader"),
                "cfg_override": self.has("CFGOverride"),
                "aura_shift": self.has("ModelSamplingAuraFlow"),
                "flux2_latent": self.has("EmptyFlux2LatentImage")}

    # ------------------------------------------------------------------ #
    # picking files
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick(names: list[str], wanted: str, contains: list[str],
              avoid: list[str] | None = None) -> str:
        if wanted and wanted in names:
            return wanted
        for n in names:
            low = n.lower()
            if all(c in low for c in contains) and \
                    not any(a in low for a in (avoid or [])):
                return n
        return ""

    # Longest first, so nvfp4 is not read as fp4.
    PRECISION_TAGS = ("nvfp4", "int8", "int4", "fp16", "bf16", "fp8", "fp4")

    @classmethod
    def _precision(cls, name: str) -> str:
        low = name.lower()
        for tag in cls.PRECISION_TAGS:
            if tag in low:
                return tag
        return ""

    def resolve_models(self, p: dict) -> dict:
        """Work out which files to load, preferring what the person chose."""
        unets, clips, vaes = self.unets(), self.clips(), self.vaes()
        if not unets:
            raise ComfyError("No files in ComfyUI/models/diffusion_models. "
                             "Download the Ideogram 4 weights on the Models page.")
        cond = self._pick(unets, p.get("cond", ""), ["ideogram4"],
                          ["unconditional", "uncond"])
        uncond = self._pick(unets, p.get("uncond", ""), ["ideogram4", "uncond"])
        if not cond:
            raise ComfyError("No Ideogram 4 model found in diffusion_models.")
        if not uncond:
            raise ComfyError(
                "The unconditional Ideogram 4 model is missing. Ideogram 4 needs "
                "both halves of the pair — download "
                "ideogram4_unconditional_*.safetensors on the Models page.")
        # The halves have to be the same precision. An fp8 conditional next to
        # an nvfp4 unconditional loads without a murmur and generates nonsense —
        # the same silent failure as running a single-model graph. Only judged
        # when both names actually carry a precision tag.
        want, got = self._precision(cond), self._precision(uncond)
        if want and got and want != got:
            partner = next((n for n in unets if "uncond" in n.lower()
                            and self._precision(n) == want), "")
            if not partner:
                raise ComfyError(
                    f"'{cond}' is {want} but the only unconditional model on "
                    f"disk is {got}. Ideogram 4 needs a matched pair — download "
                    f"the {want} unconditional half on the Models page, or "
                    f"switch to the {got} set.")
            uncond = partner
        clip = self._pick(clips, p.get("clip", ""), ["qwen3vl"]) or \
            (clips[0] if clips else "")
        if not clip:
            raise ComfyError("No text encoder found. Download "
                             "qwen3vl_8b_*.safetensors on the Models page.")
        vae = self._pick(vaes, p.get("vae", ""), ["flux2"]) or (vaes[0] if vaes else "")
        if not vae:
            raise ComfyError("No VAE found. Download flux2-vae.safetensors on "
                             "the Models page.")
        return {"cond": cond, "uncond": uncond, "clip": clip, "vae": vae}

    # ------------------------------------------------------------------ #
    # graph building
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match(available: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        low = {k.lower(): k for k in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _node(self, class_type: str, wanted: dict) -> dict:
        spec = self.node_inputs(class_type)
        inputs: dict = {}
        for key, want in wanted.items():
            name = self._match(spec, want["names"])
            if name is None:
                if want.get("required"):
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This version of "
                        "the node does not match Ideogram Studio — update it "
                        "from the Engine page.")
                continue
            inputs[name] = want["value"]
        for name, definition in spec.items():
            if name in inputs or name == "control_after_generate":
                continue
            if not isinstance(definition, (list, tuple)) or not definition:
                continue
            kind = definition[0]
            opts = definition[1] if len(definition) > 1 else {}
            if not isinstance(opts, dict):
                opts = {}
            if isinstance(kind, list):
                inputs[name] = opts.get("default", kind[0] if kind else "")
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if "default" in opts:
                    inputs[name] = opts["default"]
                elif kind == "STRING":
                    inputs[name] = ""
        return {"class_type": class_type, "inputs": inputs}

    @staticmethod
    def plain_prompt(p: dict) -> str:
        """The prompt as text, for when the KJ builder is not installed.
        Same pieces, joined in the order the builder uses them."""
        bits = [(p.get("description") or "").strip()]
        for key in ("background", "style", "style_photo", "aesthetics",
                    "lighting", "medium"):
            v = (p.get(key) or "").strip()
            if v:
                bits.append(v)
        for b in p.get("regions") or []:
            desc = (b.get("desc") or "").strip()
            text = (b.get("text") or "").strip()
            if text:
                bits.append(f'text reading "{text}"')
            elif desc:
                bits.append(desc)
        return ", ".join([b for b in bits if b])

    def build(self, p: dict) -> dict:
        """p: description, background, style, style_photo, aesthetics, lighting,
        medium, regions[], width, height, batch, steps, cfg, guider_cfg, shift,
        sampler, scheduler, denoise, seed, cond, uncond, clip, vae, filename."""
        self.ensure_supported()
        files = self.resolve_models(p)
        seed = int(p.get("seed") if p.get("seed") not in (None, "")
                   else random.randint(0, 2**40))
        width = int(p.get("width") or 1024)
        height = int(p.get("height") or 1024)
        g: dict = {}

        # loaders
        g["1"] = self._node("UNETLoader", {
            "unet": {"names": ["unet_name"], "value": files["cond"], "required": True},
            "dtype": {"names": ["weight_dtype"], "value": p.get("weight_dtype") or "default"},
        })
        g["2"] = self._node("UNETLoader", {
            "unet": {"names": ["unet_name"], "value": files["uncond"], "required": True},
            "dtype": {"names": ["weight_dtype"], "value": p.get("weight_dtype") or "default"},
        })
        clip_wanted = {
            "clip": {"names": ["clip_name"], "value": files["clip"], "required": True},
        }
        types = self.clip_types()
        if types:
            clip_wanted["type"] = {"names": ["type"],
                                   "value": "ideogram4" if "ideogram4" in types
                                   else types[0]}
        g["3"] = self._node("CLIPLoader", clip_wanted)
        g["4"] = self._node("VAELoader", {
            "vae": {"names": ["vae_name"], "value": files["vae"], "required": True},
        })

        # prompt
        text_source: object
        if self.has(PROMPT_BUILDER) and p.get("use_builder", True):
            g["5"] = self._node(PROMPT_BUILDER, {
                "width": {"names": ["width"], "value": width},
                "height": {"names": ["height"], "value": height},
                "description": {"names": ["high_level_description", "description"],
                                "value": p.get("description", ""), "required": True},
                "background": {"names": ["background"], "value": p.get("background", "")},
                "style": {"names": ["style"], "value": p.get("style", "")},
                "style_photo": {"names": ["style.photo", "style_photo"],
                                "value": p.get("style_photo", "")},
                "aesthetics": {"names": ["aesthetics"], "value": p.get("aesthetics", "")},
                "lighting": {"names": ["lighting"], "value": p.get("lighting", "")},
                "medium": {"names": ["medium"], "value": p.get("medium", "")},
                "bboxes": {"names": ["bboxes"],
                           "value": json.dumps(p.get("regions") or [])},
            })
            text_source = ["5", 0]
        else:
            text_source = self.plain_prompt(p)

        g["6"] = self._node("CLIPTextEncode", {
            "clip": {"names": ["clip"], "value": ["3", 0], "required": True},
            "text": {"names": ["text"], "value": text_source, "required": True},
        })
        g["7"] = self._node("ConditioningZeroOut", {
            "conditioning": {"names": ["conditioning"], "value": ["6", 0],
                             "required": True},
        })

        # LoRAs stack onto the conditional model only. The unconditional half is
        # the negative branch of DualModelGuider — applying the same LoRA there
        # would cancel most of its effect.
        model_ref: list = ["1", 0]
        loras = [l for l in (p.get("loras") or [])
                 if (l.get("name") or "").strip()]
        if loras:
            available = self.loras()
            cls = "LoraLoaderModelOnly" if self.has("LoraLoaderModelOnly") \
                else "LoraLoader" if self.has("LoraLoader") else ""
            if not cls:
                raise ComfyError(
                    "This ComfyUI has no LoRA loader node, so a LoRA cannot be "
                    "applied. Update ComfyUI from the Engine page.")
            for index, lora in enumerate(loras):
                name = lora["name"]
                if available and name not in available:
                    raise ComfyError(
                        f"ComfyUI cannot see the LoRA '{name}'. It may have just "
                        "been downloaded — restart ComfyUI so it rescans the "
                        "folder.")
                node_id = str(30 + index)
                wanted = {
                    "model": {"names": ["model"], "value": model_ref,
                              "required": True},
                    "lora": {"names": ["lora_name"], "value": name,
                             "required": True},
                    "strength": {"names": ["strength_model", "strength"],
                                 "value": float(lora.get("strength",
                                                          LORA_DEFAULT_STRENGTH))},
                }
                if cls == "LoraLoader":
                    wanted["clip"] = {"names": ["clip"], "value": ["3", 0]}
                    wanted["clip_strength"] = {"names": ["strength_clip"],
                                               "value": 0.0}
                g[node_id] = self._node(cls, wanted)
                model_ref = [node_id, 0]

        # model chain: shift, then the cfg window, both optional
        if self.has("ModelSamplingAuraFlow"):
            g["8"] = self._node("ModelSamplingAuraFlow", {
                "model": {"names": ["model"], "value": model_ref, "required": True},
                "shift": {"names": ["shift"], "value": float(p.get("shift") or 5.0)},
            })
            model_ref = ["8", 0]
        sched_model = model_ref
        if self.has("CFGOverride"):
            g["9"] = self._node("CFGOverride", {
                "model": {"names": ["model"], "value": model_ref, "required": True},
                "cfg": {"names": ["cfg"], "value": float(p.get("cfg") or 3.0)},
                "start": {"names": ["start_percent"],
                          "value": float(p.get("cfg_start", 0.9))},
                "end": {"names": ["end_percent"], "value": float(p.get("cfg_end", 1.0))},
            })
            model_ref = ["9", 0]

        g["10"] = self._node("DualModelGuider", {
            "model": {"names": ["model"], "value": model_ref, "required": True},
            "positive": {"names": ["positive"], "value": ["6", 0], "required": True},
            "model_negative": {"names": ["model_negative"], "value": ["2", 0],
                               "required": True},
            "negative": {"names": ["negative"], "value": ["7", 0], "required": True},
            "cfg": {"names": ["cfg"], "value": float(p.get("guider_cfg") or 7.0)},
        })

        g["11"] = self._node("RandomNoise", {
            "seed": {"names": ["noise_seed", "seed"], "value": seed, "required": True},
        })
        g["12"] = self._node("KSamplerSelect", {
            "sampler": {"names": ["sampler_name"], "value": p.get("sampler") or "euler",
                        "required": True},
        })
        g["13"] = self._node("BasicScheduler", {
            "model": {"names": ["model"], "value": sched_model, "required": True},
            "scheduler": {"names": ["scheduler"], "value": p.get("scheduler") or "simple"},
            "steps": {"names": ["steps"], "value": int(p.get("steps") or 28)},
            "denoise": {"names": ["denoise"], "value": float(p.get("denoise") or 1.0)},
        })

        latent_class = "EmptyFlux2LatentImage" if self.has("EmptyFlux2LatentImage") \
            else "EmptySD3LatentImage" if self.has("EmptySD3LatentImage") \
            else "EmptyLatentImage"
        g["14"] = self._node(latent_class, {
            "width": {"names": ["width"], "value": width, "required": True},
            "height": {"names": ["height"], "value": height, "required": True},
            "batch": {"names": ["batch_size"], "value": int(p.get("batch") or 1)},
        })

        g["15"] = self._node("SamplerCustomAdvanced", {
            "noise": {"names": ["noise"], "value": ["11", 0], "required": True},
            "guider": {"names": ["guider"], "value": ["10", 0], "required": True},
            "sampler": {"names": ["sampler"], "value": ["12", 0], "required": True},
            "sigmas": {"names": ["sigmas"], "value": ["13", 0], "required": True},
            "latent": {"names": ["latent_image"], "value": ["14", 0], "required": True},
        })
        g["16"] = self._node("VAEDecode", {
            "samples": {"names": ["samples"], "value": ["15", 0], "required": True},
            "vae": {"names": ["vae"], "value": ["4", 0], "required": True},
        })
        g["17"] = self._node("SaveImage", {
            "images": {"names": ["images"], "value": ["16", 0], "required": True},
            "prefix": {"names": ["filename_prefix"], "value": "ideogram"},
        })

        return {"prompt": g, "seed": seed, "files": files, "latent": latent_class,
                "loras": loras,
                "builder": self.has(PROMPT_BUILDER) and p.get("use_builder", True)}

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(f"{self.url}/prompt", json=body, timeout=60)
        if r.status_code >= 400:
            try:
                raise ComfyError(_readable(r.json()))
            except ValueError:
                raise ComfyError(r.text[:400])
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    def running_ids(self) -> list[str]:
        """The prompt ids ComfyUI is executing right now."""
        try:
            r = requests.get(f"{self.url}/queue", timeout=10)
            r.raise_for_status()
            entries = r.json().get("queue_running") or []
        except Exception:
            return []
        out = []
        for entry in entries:
            # [number, prompt_id, prompt, extra_data, outputs]
            if isinstance(entry, (list, tuple)) and len(entry) > 1:
                out.append(str(entry[1]))
        return out

    def cancel(self, prompt_id: str) -> None:
        """Cancel one prompt without disturbing the rest of the queue.

        /interrupt stops whatever is executing, which is not necessarily this
        prompt — several jobs can be in flight at once. So drop it from the
        queue first, and only interrupt if it is the one actually running.
        """
        try:
            requests.post(f"{self.url}/queue", json={"delete": [prompt_id]},
                          timeout=10)
        except Exception:
            pass
        if prompt_id in self.running_ids():
            self.interrupt()

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=20)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    def outputs(self, prompt_id: str) -> list[dict]:
        hist = self.history(prompt_id)
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for item in node_out.get("images", []) or []:
                if isinstance(item, dict) and item.get("filename") \
                        and item.get("type") != "temp":
                    found.append(item)
        return found

    def failed(self, prompt_id: str) -> str | None:
        status = (self.history(prompt_id).get("status") or {})
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    return (f"{data.get('node_type')}: "
                            f"{data.get('exception_message')}")
            return "ComfyUI reported an error while generating."
        return None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return requests.get(f"{self.url}/view", params=params, stream=True,
                            timeout=180)

    def upload_image(self, file_storage) -> str:
        files = {"image": (file_storage.filename, file_storage.stream,
                           file_storage.mimetype or "image/png")}
        r = requests.post(f"{self.url}/upload/image", files=files,
                          data={"type": "input", "overwrite": "true"}, timeout=180)
        r.raise_for_status()
        data = r.json()
        name = data.get("name") or file_storage.filename
        sub = data.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name


def _readable(err: dict) -> str:
    for node_id, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            return (f"{info.get('class_type', 'node ' + str(node_id))}: "
                    f"{e.get('message')} {e.get('details', '')}".strip())
    top = err.get("error") or {}
    if top:
        return f"{top.get('message', 'Rejected by ComfyUI')} " \
               f"{top.get('details', '')}".strip()
    return json.dumps(err)[:300]
