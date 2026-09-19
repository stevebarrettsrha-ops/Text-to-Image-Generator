# Ideogram Studio — invariants

## Hard rules (do not revisit)

1. **`web/index.html` stays one file with no build step.** This one deliberately
   does NOT follow the YuE Studio / Script Builder shell: image tools put a prompt
   bar over a masonry feed, with settings in popovers off the bar, so that is what
   it does. Keep the rail and the Models/Engine pages consistent with the other
   two; leave the Generate page alone.
1b. **A hidden page has no layout.** `scrollHeight` is 0 and elements are not
   clickable inside a hidden section or a closed `<details>`, so anything that
   measures (textarea auto-grow, canvas sizing) runs on `show()` via
   `requestAnimationFrame`, and Playwright selectors must be scoped to the visible
   page (`#feed .tile`, not `.tile`).
2. **Never hard-code a ComfyUI workflow.** `comfy.py` builds the graph from
   `/object_info` and matches inputs through candidate-name lists.
3. **Ideogram 4 is a pair.** The conditional model feeds the chain and the
   unconditional one feeds `DualModelGuider.model_negative`. Never substitute one
   for the other, and fail loudly when only one is on disk — a silent single-model
   graph produces nonsense.
4. **`BasicScheduler` takes the model from `ModelSamplingAuraFlow`, not from
   `CFGOverride`.** That is how ideogram4.json is wired (links 292 vs 291);
   sigmas must be computed before the CFG window is applied.
5. **Custom node requirements install into the interpreter ComfyUI runs on** —
   portable `python_embeded\python.exe` first, then the managed venv, never the
   system Python.
6. **Python detection is by execution, never PATH lookup.** Windows Store stubs
   resolve on PATH and fail to run.
7. **Downloads are resumable**: `.part` file, `Range` on retry, atomic replace.
8. **Model deletes are path-checked**: folder must be in `MODEL_FOLDERS`, no
   separators or `..` in the name, resolved path under `models_dir`.
9. **Region coordinates stay normalised 0–1** with the keys the builder expects:
   `x, y, w, h, type, text, desc, palette`. Do not switch to pixels — a layout has
   to survive an aspect-ratio change.

## Nodes the workflow uses that this app does not

`ideogram4.json` also carries `Seed (rgthree)`, `FluxResolutionNode`,
`easy showAnything`, `SigmasPreview` and `PreviewAny`. They are canvas
conveniences — seed, resolution maths and previews — which the front end does
itself, so their packs are deliberately **not** installed. Only ComfyUI-Manager
and KJNodes are. If a future change needs one, add it to `CUSTOM_NODES` in
bootstrap.py; the installer and the Engine page pick it up automatically.

## Reading /object_info

ComfyUI's V3 node API serialises every input as `(io_type, options)`, so a
dropdown arrives as `("COMBO", {"options": [...]})` and a dynamic one as
`("COMFY_DYNAMICCOMBO_V3", {"options": [{"key", "inputs"}]})` — not the old
`([...options], {...})`. `_node()` fills any input it was not asked for from the
schema default, and it has to understand all three shapes or the input is left
out and ComfyUI answers "Required input is missing". It also skips anything
marked `forceInput`: that is a link, and there is no widget value to give it.

`Ideogram4PromptBuilderKJ` is now a V3 node, which moved two inputs this app
fills:

- **`style` is a dynamic combo.** The prompt carries the chosen key
  (`none` / `photo` / `art_style`) under `style`, and the text for that branch
  under the dotted path the expansion creates — `style.photo`,
  `style.art_style`. Those nested ids are not in `/object_info` until a key is
  picked, so they are set by hand rather than matched. The UI's style word and
  style detail map onto it: "photo" takes the photo branch, anything else is an
  art style, neither is `none`.
- **Regions go in `elements_data`**, as the same normalised list the editor
  keeps. `bboxes` is a link-only BOUNDINGBOX input now (pixel-space, and only
  ever a seed), so writing region JSON there silently loses placement. Older
  builds took that JSON on `bboxes` as a string, so the shape in `/object_info`
  decides which one to use, never a version number.

## Graceful degradation

Without KJNodes there is no `Ideogram4PromptBuilderKJ`, so `build()` falls back to
`plain_prompt()` — the fields and regions joined into one string. It generates,
but placement is lost. Keep the fallback honest: the UI says so rather than
pretending regions worked. `CFGOverride` and `ModelSamplingAuraFlow` are skipped
the same way if absent.

## LoRA rules (from the DeverStyle model card, not guesses)

- Strength default **0.6**, applied to the **conditional model only**. Never
  chain the LoRA onto the unconditional model feeding
  `DualModelGuider.model_negative`.
- Trigger words live in the **style** section of the structured prompt, and are
  parsed from the brackets in the file name:
  `dever_arcane_style_ideogram4 (arcvfx).safetensors` -> `arcvfx`.
- **No in-app LoRA browser and no site proxy.** One was built and removed on
  purpose: framing Civitai or HuggingFace means stripping their
  X-Frame-Options/CSP through a local proxy, which is fragile against their
  JavaScript and is an open-relay shape. The Models page's repo browser covers
  the same need with code that is already tested. Do not reintroduce it.
- ComfyUI enumerates `models/loras` at startup only, so validate the chosen LoRA
  against the live `/object_info` enum and tell the person to restart rather than
  letting the graph fail.

## Progress reporting

The first run is long and mostly silent from the outside, so every step says
where it is. `Progress.track(key, pct, detail)` moves one step's bar; `pct=None`
means "running, no number to give" and draws an indeterminate bar rather than a
fake 0%. Sources of a real number:

- **git** — `git_run`/`git_clone` pass `--progress` and `_stream` splits on `\r`,
  so the phase lines arrive while the clone runs. `GIT_WEIGHT` stacks the phases
  into one bar that only moves forwards, and git runs under `LC_ALL=C` because
  the phase names are matched in English.
- **pip** — its bar disappears when stdout is a pipe, which is what made a
  2.4 GB torch wheel look like a hang. `--progress-bar raw` prints
  `Progress <done> of <total>` instead; `pip_has_raw_progress` asks pip whether
  it has the option (older ones do not) and the answer is dropped after pip
  upgrades itself.
- **downloads** — the weights bar is over the whole set, so `hf_tree` is asked
  for the real sizes first. Falling back to per-file percentages is fine; a bar
  that restarts four times is not.

`snapshot()` sends the steps as an **ordered list**, not a dict: Flask sorts JSON
object keys, which served them alphabetically and put "Check Python" last.
`app.json.sort_keys = False` covers the rest of the API.

## Validation gate — run after any edit

```bash
python -m py_compile server.py comfy.py bootstrap.py manager.py
python - <<'PY'
import re, pathlib
src = pathlib.Path('web/index.html').read_text()
pathlib.Path('/tmp/ig.js').write_text('\n'.join(re.findall(r'<script>(.*?)</script>', src, re.S)))
PY
node --check /tmp/ig.js
```

Note when testing with Playwright: `inner_text` returns "" for anything inside a
closed `<details>`. Use `text_content` for values in collapsed cards.
