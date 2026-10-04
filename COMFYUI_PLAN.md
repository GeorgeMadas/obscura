# PLAN: add ComfyUI to Obscura (custom workflows + video)

**Audience: Claude Code on the new, larger Vast instance.** Do SETUP.md first (Obscura working),
then this plan, phase by phase. Each phase ends with a **Check**. Don't move on from a failing check.
Written 2026-10-02.

Ground rules (same as SETUP.md):
- **Never trigger a render, upscale or ComfyUI workflow run to test anything.** The user does all
  testing. Verify with syncing, `py_compile`, `node --check`, port/log checks, and read-only
  ComfyUI endpoints (`/system_stats`, `/object_info`, `/queue`).
- Don't touch the NVIDIA driver or reinstall torch (Blackwell, cc 12.0, needs cu128+).
- After each phase that changes the app, commit and push to `GeorgeMadas/obscura`.

## Goal

1. Install ComfyUI on the instance as a managed service, and keep its editor available for building
   workflows.
2. Obscura gets a **Workflows** tab. The user can drop in any ComfyUI workflow (API format) and
   run it from a simple form. Results go into the normal Library and history.
3. **Video:** Wan 2.2 image-to-video and motion-transfer (Wan 2.2 Animate) presets, video playback
   in the Library and viewer, and an "Animate" button on any image. **No text-to-video** (user
   decision 2026-10-04).

## Key facts that shape the design

| Fact | Consequence |
|---|---|
| Obscura keeps Qwen fully on the GPU: **~88 GB of 96 GB VRAM** while idle | ComfyUI can't run alongside it. VRAM has to be handed back and forth (phase 3). |
| System RAM is ~1.5 TB | Parking Qwen in CPU RAM is cheap; moving it back takes seconds, not a reload from disk. |
| Obscura already serializes GPU work with `GPU_LOCK` and a single `_worker` queue | ComfyUI jobs go into the **same queue** as a new `kind`, so a job never overlaps a Qwen render. |
| The old instance had 24 GB free on 100 GB | New instance: **≥ 250 GB disk** (Qwen ~75 GB + Wan 14B ~40 GB + ComfyUI/nodes + outputs + headroom). |

## Phase 0: preflight

- `df -h /workspace` shows ≥ 150 GB free after SETUP.md is done.
- `vast-capabilities | jq '.instance.open_ports[]|select(.in_use==false)'` shows a free **normal**
  port for the ComfyUI editor. The old instance had `10200`. If there isn't one, skip external
  exposure and use SSH forwarding (§7 of CLAUDE.md).

**Check:** enough disk, and a port chosen (or SSH-only decided).

## Phase 1: install ComfyUI

- `git clone https://github.com/comfyanonymous/ComfyUI /workspace/ComfyUI`
- Install its `requirements.txt` into `/venv/main` with `uv pip`. **Exclude torch, torchvision and
  torchaudio** so the cu128 torch stays. Afterwards confirm the torch version is unchanged.
- Custom nodes in `/workspace/ComfyUI/custom_nodes/`:
  - `ComfyUI-Manager` (lets the user install nodes from the UI)
  - `ComfyUI-VideoHelperSuite` (video load/combine → mp4/webm)
  - `ComfyUI-KJNodes` (common helpers in Wan workflows)
  - Install each one's `requirements.txt`, again without touching torch.
- Possible conflict: ComfyUI and Obscura share `/venv/main`. If ComfyUI needs a version of
  `transformers`, `diffusers` or other packages that breaks Obscura, give ComfyUI **its own venv**
  (`/venv/comfy`, with the same cu128 torch wheel) instead. Decide based on what pip reports.

**Check:** `python main.py --help` runs, and `python -m py_compile /workspace/qwen-image/app.py`
still passes; Obscura's own imports still load.

## Phase 2: ComfyUI as a supervisor service

- `/opt/supervisor-scripts/comfyui.sh`, following the pattern in CLAUDE.md §7:
  `exit_portal.sh "ComfyUI"`, then
  `pty python main.py --listen 127.0.0.1 --port 18188 --output-directory /workspace/ComfyUI/output`.
  Add `--disable-auto-launch`.
- `/etc/supervisor/conf.d/comfyui.conf`, modeled on `tensorboard.conf`.
- Portal entry `ComfyUI`: external `10200` → internal `18188`, behind Caddy token auth. Restart caddy.
- Save copies of the script and conf in `deploy/service/` in the repo, like the qwen-image ones.

**Check:** `supervisorctl status comfyui` shows RUNNING, `curl -s 127.0.0.1:18188/system_stats`
returns JSON, and the editor URL loads with the token.

## Phase 3: VRAM handoff (most important)

Without this, any ComfyUI video job runs out of memory while Qwen sits on 88 GB.

In `app.py`:
- `park_qwen()`: under `GPU_LOCK`, move the pipeline (transformer, text encoder, VAE, loaded LoRAs)
  to CPU and call `torch.cuda.empty_cache()`. Set a `QWEN_ON_GPU` flag.
- `unpark_qwen()`: move it back to `cuda`.
- Everything that uses the pipe (`run_generation`, `run_enhance`, rewrite if it uses the GPU, model
  switch) calls `unpark_qwen()` first when the flag is off. Upscale (ESRGAN) is small and can stay.
- In the other direction, before a ComfyUI job: `park_qwen()`. After it finishes: ask ComfyUI to free
  its own VRAM (`POST /free {"unload_models": true, "free_memory": true}`). Leave Qwen parked; it
  comes back on the next image render. Show "Loading Qwen back to GPU" as the job phase so the user
  knows why the first render after a video takes a few extra seconds.
- Settings toggle (optional): "Keep ComfyUI models loaded between runs". That's faster for
  back-to-back videos but keeps Qwen parked.

**Check:** read the code paths again. Every pipe use is preceded by unpark, and park/unpark only run
inside `GPU_LOCK`. No test render; the user tests.

## Phase 4: ComfyUI bridge in the backend

New module `comfy_bridge.py`, used by `app.py`:
- `COMFY_URL = http://127.0.0.1:18188`, overridable by env.
- **Workflow store:** `/workspace/qwen-image/workflows/<slug>.json`, holding the API-format graph
  plus a small sidecar `<slug>.meta.json` (name, description, output type image/video, exposed
  inputs). Add `workflows/` to git so presets sync between instances.
- **Input detection** when a workflow is imported: scan nodes and expose the obvious knobs as form
  fields:
  - positive/negative prompt (`CLIPTextEncode` and similar text inputs)
  - seed (`KSampler*`, `RandomNoise`, with a randomize option)
  - steps, cfg, width/height/length (frames), fps
  - `LoadImage` → image upload or "pick from Library"
  - The user can rename, hide or add fields in an "Edit fields" dialog. The mapping is
    `node_id.input_name → label/type/default`.
- **Running a job** = a new queue `kind: "comfy"` in `_worker`:
  1. `park_qwen()`
  2. Upload any input images to ComfyUI (`POST /upload/image`).
  3. Patch the graph with the form values and `POST /prompt` with a `client_id`.
  4. Follow `ws://…/ws?clientId=…` for `progress` / `executing` / `executed` and map them to the
     existing job `phase`/progress fields, so the current progress UI just works.
  5. Cancel: `POST /interrupt`, then remove the job from the ComfyUI queue.
  6. When done: `GET /history/{prompt_id}`, then fetch each output with `/view`, copy it into
     `OUTPUT_DIR`, and write a normal history record (`kind: "comfy"`, workflow slug, the form
     values, `media: "video"|"image"`).
  7. Free ComfyUI VRAM (phase 3).
- If ComfyUI is down, return a clear `UserError` ("ComfyUI isn't running") instead of a stack trace.
- New routes:
  - `GET /api/workflows`
  - `POST /api/workflows` (import JSON; rejects UI-format JSON with a "use Export (API)" hint)
  - `PUT /api/workflows/{slug}` (field mapping)
  - `DELETE /api/workflows/{slug}`
  - `POST /api/workflows/{slug}/run`
  - `GET /api/comfy/status` (up/down, VRAM, queue)

**Check:** `py_compile` on both files, the service restarts cleanly, `/api/workflows` and
`/api/comfy/status` respond.

## Phase 5: video support in the app

- History and Library: `_checked_output`, thumbnails and `/api/history` handle `.mp4`/`.webm`.
  Thumbnail = first frame, extracted with `ffmpeg` (check that it's installed; `apt install ffmpeg`
  if not) into `.thumbs/`.
- Library grid: video tiles show a ▶ badge and duration, and play muted on hover.
- Viewer: `<video controls loop>`, with frame count, fps and resolution in the info panel.
- "Delete", "related" and the other per-item actions work for videos too.
- The **Animate** button on any image opens the Wan I2V preset with that image already filled in.

**Check:** `node --check static/app.js`. The user tests the UI.

## Phase 6: Workflows tab (frontend)

- New tab beside the existing ones, in the same style (the Obscura look in `styles.css`).
- Left: workflow list (presets + imported), with an Import button (paste or drop an API JSON).
- Right: a form generated from the field mapping, a Run button, and the shared progress bar and
  queue. Results appear in the Library.
- "Open in ComfyUI editor" link (port 10200 URL) for building or changing workflows. Short tip:
  *ComfyUI → Workflow → Export (API)* to get the right JSON.

**Check:** `node --check static/app.js`; the service serves the new tab.

## Phase 7: models + presets

Put models under `/workspace/ComfyUI/models/…`. Download with `hf download`, the same way SETUP.md
step 5 does. Pick the current repack repos/filenames at install time; ComfyUI's official Wan 2.2
example workflows list the exact files.

User decisions (2026-10-04): **no T2V 14B**. Install I2V 14B, the 5B draft model, and the
motion-transfer model (Wan 2.2 Animate 14B).

| Use | Files (approx. size) |
|---|---|
| Wan 2.2 I2V 14B | high-noise + low-noise, fp8 (~2 × 14 GB) |
| Wan 2.2 Animate 14B (motion transfer) | diffusion model fp8 (~17 GB), plus `clip_vision_h` (~1.3 GB) and its relight LoRA |
| Wan 2.2 TI2V 5B (quick drafts) | diffusion model (~10 GB) + **Wan 2.2 VAE** (5B uses the 2.2 VAE, not 2.1) |
| Shared | umt5-xxl text encoder fp8 (~7 GB), Wan 2.1 VAE (~0.3 GB) |
| Optional, fast | Lightx2v / Lightning 4-step LoRAs for Wan 2.2 I2V (~1 GB) |

Motion transfer also needs pose/face preprocessing nodes (e.g. kijai's
`ComfyUI-WanAnimatePreprocess` or whatever ComfyUI's official Wan 2.2 Animate example uses; check
at install time and add them in phase 1).

Ship presets in `workflows/`: **Wan 2.2 Image → Video** (14B, plus a fast 4-step version if the
LoRAs are installed), **Wan 2.2 Draft Image → Video (5B)**, and **Wan 2.2 Motion Transfer**
(inputs: a character image + a driving video). Fields: prompt, negative, seed, width/height,
length (frames), fps, and steps. Motion transfer also needs a **video input field** (upload or pick
a video from the Library), so the bridge's input detection must handle video-load nodes
(e.g. VHS `LoadVideo`) as well as `LoadImage`.

Add the downloads to `deploy/prefetch_models.py` (or a new `deploy/prefetch_comfy.py`).

**Check:** files present, sizes sane; `GET /object_info` lists the nodes the presets use. No run.

## Phase 8: keep the instances reproducible

- Add a **"9. ComfyUI"** section to SETUP.md covering the clone, nodes, service, portal entry, models
  and the `/venv/comfy` decision. Update the snapshot table (disk ≥ 250 GB, port 10200).
- Commit and push. Tell the user exactly what to test:
  1. The ComfyUI editor opens with the token.
  2. Import one of their own workflows and run it.
  3. Wan I2V run (14B and 5B draft).
  4. Motion transfer: a character image + a driving video.
  5. Animate an existing image.
  6. A Qwen render right after a video (checks the VRAM handoff).

## User decisions (answered 2026-10-04)

- Models: **I2V 14B + 5B draft + Wan 2.2 Animate (motion transfer)**. No T2V.
- **Expose the ComfyUI editor on a port** (10200 → 18188, behind Caddy token auth).
- Up-front nodes: whatever the motion-transfer workflow needs (Wan Animate preprocess nodes).
