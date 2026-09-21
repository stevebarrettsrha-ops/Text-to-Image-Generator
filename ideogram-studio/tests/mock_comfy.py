"""A stand-in ComfyUI: /object_info and /system_stats shaped like the real
thing, for the tests that are about the engine rather than the graph.

The node classes and input names are the ones comfy.py actually reads —
UNETLoader.unet_name is the list resolve_models() refuses without, and it is
served in the old `([options], {...})` shape that ComfyUI still uses for model
dropdowns, because that is the shape `_enum()` parses.

The model lists are a **startup scan**, like the real server's: they are read
once, when this process starts, and a file that lands afterwards is invisible
until it is restarted. That is the whole bug the engine kit is about.

Knobs, all environment variables:
  MOCK_MODELS_DIR    scan this folder for the model lists (default: the
                     built-in fp8 set, as though the weights were there)
  MOCK_BLANK_UNETS   serve an empty diffusion-model list — an engine that
                     started before the weights landed
  MOCK_COMFY_ROOT    the install dir /system_stats claims via argv
                     (default none: no argv, like wrappers that hide it)
"""
import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_MODELS = {
    "diffusion_models": ["ideogram4_fp8_scaled.safetensors",
                         "ideogram4_unconditional_fp8_scaled.safetensors"],
    "text_encoders": ["qwen3vl_8b_fp8_scaled.safetensors"],
    "vae": ["flux2-vae.safetensors"],
    "loras": [],
}


def _scan() -> dict:
    """What this engine found in its model folders — once, at startup."""
    found = dict(DEFAULT_MODELS)
    root = os.environ.get("MOCK_MODELS_DIR")
    if root:
        base = pathlib.Path(root)
        found = {folder: sorted(p.name for p in (base / folder).glob("*")
                                if p.is_file())
                 for folder in DEFAULT_MODELS}
    if os.environ.get("MOCK_BLANK_UNETS"):
        found["diffusion_models"] = []
    return found


MODELS = _scan()
LOCK = threading.Lock()
UPLOADS: list = []          # filenames handed to /upload/image


def _object_info() -> dict:
    """The schema, with the startup scan in the dropdowns and anything
    uploaded since visible to LoadImage — the real server rescans its input
    folder but not its model folders."""
    with LOCK:
        uploads = list(UPLOADS)

    def combo(values):
        """A model dropdown, in the shape ComfyUI serves them."""
        return [list(values), {}]

    def node(required=None, optional=None):
        spec = {"required": dict(required or {})}
        if optional:
            spec["optional"] = dict(optional)
        return {"input": spec, "output": [], "output_name": []}

    info = {
        "UNETLoader": node({
            "unet_name": combo(MODELS["diffusion_models"]),
            "weight_dtype": combo(["default", "fp8_e4m3fn"])}),
        "CLIPLoader": node({
            "clip_name": combo(MODELS["text_encoders"]),
            "type": combo(["qwen3vl", "flux2", "stable_diffusion"])}),
        "VAELoader": node({"vae_name": combo(MODELS["vae"])}),
        "LoraLoaderModelOnly": node({
            "model": ["MODEL", {}],
            "lora_name": combo(MODELS["loras"]),
            "strength_model": ["FLOAT", {"default": 1.0}]}),
        "KSamplerSelect": node({
            "sampler_name": combo(["euler", "dpmpp_2m", "res_multistep"])}),
        "BasicScheduler": node({
            "model": ["MODEL", {}],
            "scheduler": combo(["simple", "normal", "beta"]),
            "steps": ["INT", {"default": 20}],
            "denoise": ["FLOAT", {"default": 1.0}]}),
        "CLIPTextEncode": node({"text": ["STRING", {"multiline": True}],
                                "clip": ["CLIP", {}]}),
        "ConditioningZeroOut": node({"conditioning": ["CONDITIONING", {}]}),
        "DualModelGuider": node({
            "model": ["MODEL", {}], "model_negative": ["MODEL", {}],
            "conditioning": ["CONDITIONING", {}],
            "negative": ["CONDITIONING", {}],
            "cfg": ["FLOAT", {"default": 7.0}]}),
        "RandomNoise": node({"noise_seed": ["INT", {"default": 0}]}),
        "SamplerCustomAdvanced": node({
            "noise": ["NOISE", {}], "guider": ["GUIDER", {}],
            "sampler": ["SAMPLER", {}], "sigmas": ["SIGMAS", {}],
            "latent_image": ["LATENT", {}]}),
        "VAEDecode": node({"samples": ["LATENT", {}], "vae": ["VAE", {}]}),
        "SaveImage": node({"images": ["IMAGE", {}],
                           "filename_prefix": ["STRING", {"default": "ComfyUI"}]}),
        "EmptyFlux2LatentImage": node({
            "width": ["INT", {"default": 1024}],
            "height": ["INT", {"default": 1024}],
            "batch_size": ["INT", {"default": 1}]}),
        "CFGOverride": node({"model": ["MODEL", {}],
                             "cfg": ["FLOAT", {"default": 3.0}]}),
        "ModelSamplingAuraFlow": node({"model": ["MODEL", {}],
                                       "shift": ["FLOAT", {"default": 5.0}]}),
        "Ideogram4PromptBuilderKJ": node({
            "description": ["STRING", {"multiline": True}],
            "background": ["STRING", {"multiline": True}],
            "style": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "none", "inputs": {}},
                {"key": "photo", "inputs": {"photo": ["STRING", {}]}},
                {"key": "art_style", "inputs": {"art_style": ["STRING", {}]}}]}],
            "elements_data": ["STRING", {"multiline": True}]}),
        "LoadImage": node({"image": combo(uploads)}),
        "ImageScale": node({"image": ["IMAGE", {}],
                            "upscale_method": combo(["lanczos", "nearest-exact"]),
                            "width": ["INT", {}], "height": ["INT", {}],
                            "crop": combo(["disabled", "center"])}),
        "VAEEncode": node({"pixels": ["IMAGE", {}], "vae": ["VAE", {}]}),
        "RepeatLatentBatch": node({"samples": ["LATENT", {}],
                                   "amount": ["INT", {"default": 1}]}),
        "ImageToMask": node({"image": ["IMAGE", {}],
                             "channel": combo(["red", "green", "blue"])}),
        "SetLatentNoiseMask": node({"samples": ["LATENT", {}],
                                    "mask": ["MASK", {}]}),
    }
    return info


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/object_info":
            self._send(200, _object_info())
        elif p == "/system_stats":
            system = {"comfyui_version": "0.3.75"}
            root = os.environ.get("MOCK_COMFY_ROOT", "")
            if root:
                system["argv"] = [f"{root}/main.py"]
            self._send(200, {"system": system})
        elif p == "/queue":
            self._send(200, {"queue_running": [], "queue_pending": []})
        else:
            self._send(404, {"error": "no route " + p})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if p == "/upload/image":
            marker = b'filename="'
            i = raw.find(marker)
            name = raw[i + len(marker):raw.index(b'"', i + len(marker))].decode() \
                if i >= 0 else "upload.bin"
            with LOCK:
                if name not in UPLOADS:
                    UPLOADS.append(name)
            self._send(200, {"name": name, "subfolder": "", "type": "input"})
        else:
            # No ComfyUI-Manager here, so /manager/reboot is a 404 — which is
            # how a plain ComfyUI answers it, and what sends the takeover down
            # the stop-the-process path.
            self._send(404, {"error": "no route " + p})


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
