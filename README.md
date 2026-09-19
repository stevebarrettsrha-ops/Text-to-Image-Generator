# Ideogram Studio

A local image generator for **Ideogram 4**. Describe the picture, set style and
lighting as separate fields the way Ideogram expects, draw boxes for what goes
where — including text it should render — and press Generate. Everything runs on
your own GPU.

---

## Running it

**Windows** — double-click `run.bat`
**macOS / Linux** — `./run.sh`

The browser opens at <http://127.0.0.1:7802>.

On the first run a setup panel appears with three routes:

| Route | What happens |
|---|---|
| Use an existing ComfyUI | Any install found is listed. Only the nodes and weights are added. |
| Install a fresh ComfyUI | Clones ComfyUI into `./ComfyUI`, adds the nodes, builds its own Python environment, downloads the weights. |
| Connect to one I start myself | Set the address in Settings. Setup adds the nodes and weights only. |

Setup then does, in order:

1. clones **ComfyUI-Manager** and **ComfyUI-KJNodes** into `ComfyUI/custom_nodes/`
2. `pip install -r requirements.txt` for each, into the interpreter ComfyUI
   actually runs on — `python_embeded\python.exe` on a portable install, the
   virtual environment beside it otherwise
3. downloads the Ideogram 4 weights into `ComfyUI/models/`

Each step reports itself while it runs: which of the six it is on, a bar, and a
percentage with the bytes, speed and time left behind it. The weights bar covers
the whole set, not one file at a time, so it only ever moves forwards; PyTorch
reports per wheel, because pip gives no total for an install. Anything with no
number to report — creating the environment, waiting for ComfyUI to come up —
says what it is doing and how long it has been at it rather than showing a
stalled 0%.

### Requirements

- Python 3.10 or newer, and Git
- An NVIDIA card with 12 GB or more for the fp8 weights. Less means the nvfp4 or
  int8 set, which the Engine page will suggest when it sees your VRAM.

### Weights

All from [Comfy-Org/Ideogram-4](https://huggingface.co/Comfy-Org/Ideogram-4).
Ideogram 4 is a **pair** — a conditional and an unconditional model — and both
halves are needed, plus a text encoder and a VAE:

| Set | Files | Notes |
|---|---|---|
| fp8 | `ideogram4_fp8_scaled` + `ideogram4_unconditional_fp8_scaled` | ~9.3 GB each, widest support |
| int8 | `ideogram4_int8_convrot` + unconditional | RTX 30 series and newer |
| nvfp4 | `ideogram4_nvfp4_mixed` + unconditional | ~5.5 GB each, RTX 50 series |

Plus `qwen3vl_8b_fp8_scaled.safetensors` (text encoder) and
`flux2-vae.safetensors`, shared by every set. Pick a set on the Models page and
press Download set; switching later is one button.

The weights are under Ideogram's non-commercial licence. If a download comes
back refused, accept it on the HuggingFace model page and paste a token on the
Models page.

---

## Making an image

The Generate page is one prompt bar over a feed of results, the way the hosted
image tools work. Type, press Generate, and the run appears at the top of the
feed as a placeholder that fills in when it finishes.

Everything else hangs off the bar:

- **Aspect chips** — 1:1 through 16:9. Size in megapixels is under Settings.
- **Style** — opens the structured fields Ideogram 4 actually wants: style, style
  detail, lighting, medium, background, aesthetics. The pill shows how many are
  filled. Filling a few of them beats stuffing everything into one sentence.
- **Layout** — the region editor. Drag on the frame to add a region and say what
  belongs inside it; set one to **Text** for words Ideogram should render. The
  pill shows the region count. Coordinates are normalised, so a layout survives a
  change of aspect ratio.
- **Settings** — steps, prompt strength (the dual-model guider's CFG, the main
  dial), late CFG, shift, sampler, scheduler, seed.
- **Model pill** and **×N** for images per run.

Hover any image for reuse, download and delete. Click it for the full view, with
the prompt and every setting listed down the side — **Reuse settings** loads it
all back into the bar, **Vary** does the same but clears the seed and fires a new
run straight away.

`Ctrl` + `Enter` generates from anywhere.

## LoRAs

There is no LoRA browser in the app — the Models page already pulls any file
from any HuggingFace repo into any model folder. Paste a LoRA repo such as
`DeverStyle/Ideogram-4.0-Loras` into **Browse a repo**, and its files are
offered with `loras` already selected as the destination. Dropping
`.safetensors` files into `models/loras` by hand works just as well.

The **LoRAs** pill in the prompt bar lists whatever is in that folder, each with
a strength slider. Two details come from the DeverStyle model card and are built
in:

- Strength defaults to **0.6**, which is what the author's own samples use.
- The LoRA is applied to the **conditional model only**, never to
  `model_negative` — the author's instruction, and applying it to both would
  cancel most of the effect.

Where a file name carries a trigger word in brackets —
`dever_arcane_style_ideogram4 (arcvfx).safetensors` — it is shown beside the
name, and switching the LoRA on adds it to the **Style** field where these LoRAs
expect it. Switching off takes it back out.

ComfyUI scans the LoRA folder only at startup, so a freshly downloaded file
needs a ComfyUI restart before it can be used. The app checks and says so rather
than failing mid-generation.

## Engine and Models pages

**Engine** lists Python, Git, ComfyUI, each custom node, PyTorch, the weights and
the running engine, with an Install button on anything missing and
**Install everything missing** to work through them in order. It reads your VRAM
and says so when a lighter weight set would be the better idea.

**Models** holds the three weight sets with per-file status, the HuggingFace token
and mirror, a repo browser, live download progress with Stop, and delete. Downloads
resume where they stopped. All three sets are listed, but you only ever need one:
only the set you are using can be short of anything, and files belonging to the
other two read *not needed* rather than *missing*.

---

## Troubleshooting

**"Nodes not loaded"** — ComfyUI is up but has not imported the nodes. Restart it.
If it persists, check the ComfyUI console for `IMPORT FAILED`.

**Regions seem to be ignored** — KJNodes is not loaded, so there is no
`Ideogram4PromptBuilderKJ`. The app falls back to folding the fields and regions
into one text prompt, which still generates but cannot place things. Install
KJNodes from the Engine page.

**"The unconditional model is missing"** — only one half of the pair was
downloaded. Get the matching `ideogram4_unconditional_*` file.

**Out of memory** — switch to nvfp4 or int8 on the Models page, and drop the
megapixel slider.

---

## Layout

```
server.py      Flask API — image jobs, gallery, setup, dependencies, HuggingFace
bootstrap.py   Discovery, installs, weight downloads, ComfyUI process
manager.py     Dependency checks and installers, HuggingFace browsing
comfy.py       Builds the Ideogram 4 graph from ComfyUI's live schema
web/index.html The interface — one file, no build step
assets/        ideogram4_reference.json, the workflow this was built from
data/          config.json, gallery.json, images/
```

Port: set `IDEOGRAM_STUDIO_PORT`. Set `IDEOGRAM_STUDIO_NO_BROWSER=1` to stop it
opening a tab.
