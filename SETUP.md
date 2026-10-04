# SETUP: rebuild the Obscura / Qwen Image instance

**Audience: Claude Code, running on a fresh Vast.ai instance.** The user installs Claude Code
themselves and then asks you to follow this file. Do the steps in order. Every step ends with a
**Check**; do not continue past a failing check — fix it or report it to the user.

Snapshot of the reference instance this guide reproduces (2026-10-01):

| Thing | Value |
|---|---|
| Vast image | `vastai/pytorch:cuda-12.8.1-auto` (torch `2.11.0+cu128`, Python 3.12, venv `/venv/main`) |
| GPU | RTX PRO 6000 Blackwell, 96 GB (cc 12.0 → needs a cu128+ torch) |
| App dir | `/workspace/qwen-image` (this repo, remote `GeorgeMadas/obscura`, branch `main`) |
| App | FastAPI + uvicorn, `app.py`, binds `127.0.0.1:17860` |
| Service | supervisor program `qwen-image`, portal name `Qwen Image` |
| External port | `10100` → internal `17860`, behind Caddy token auth |
| Models | `$HF_HOME` = `/workspace/.hf_home` (~75 GB, list in step 5) |
| ComfyUI | `/workspace/ComfyUI`, venv `/venv/comfy`, service `comfyui` on `127.0.0.1:18188`, portal `ComfyUI` `10200` → `18188` (step 9) |
| Video models | `/workspace/ComfyUI/models` (~80 GB: Wan 2.2 I2V 14B, TI2V 5B, Wan 2.2 Animate + pose models) |
| Disk | **≥ 300 GB** (Qwen ~75 GB + video models ~80 GB + two venvs ~20 GB + outputs) |
| Open ports | the template must open **10100** (Obscura) and **10200** (ComfyUI editor) |
| Claude extras | `~/.claude/settings.json`, 13 skills, 2 memories (step 7) |

Ground rules while doing this:
- Read `/workspace/CLAUDE.md` (Vast instance guide) first if you haven't — supervisor, Caddy and
  `/etc/portal.yaml` work exactly as described there.
- **Never trigger a generation or upscale to test the app** (no `/api/generate`, `/api/upscale`,
  no clicking Generate). The user does all testing. Verify with process/port/log checks only.
- Don't touch the NVIDIA driver; don't change torch unless step 1's check fails.

---

## 0. Get the repo (bootstrap)

If you are reading this from `/workspace/qwen-image/SETUP.md`, the repo is already cloned —
skip to step 1. Otherwise the user will have given you this command:

```bash
git clone https://x-access-token:<GITHUB_TOKEN>@github.com/GeorgeMadas/obscura.git /workspace/qwen-image
```

The token lives in `deploy/github.env` inside the repo once cloned.

**Check:** `test -f /workspace/qwen-image/app.py && git -C /workspace/qwen-image log --oneline -1`

## 1. Confirm the base environment

```bash
source /venv/main/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
vast-capabilities | jq '.hardware.gpu.cuda | {compute_capability, min_cuda_for_wheels}'
vast-capabilities | jq '.instance.open_ports[] | {container_port, public_port, in_use, self_mapped}'
df -h /workspace
```

**Check:**
- `torch.cuda.is_available()` is `True` and the CUDA version ≥ `min_cuda_for_wheels` (if set).
  If not: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- At least **~100 GB free** disk and **≥ 48 GB VRAM** (the BF16 model + text encoder are large;
  with less VRAM tell the user to switch to the "Small" model in the UI).
- Note which **free, non-self-mapped** port exists for step 4 — prefer `10100`.

## 2. GitHub connection (push/pull from this instance)

```bash
cd /workspace/qwen-image
set -a; . deploy/github.env; set +a
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\n' "$GITHUB_TOKEN" > ~/.git-credentials
chmod 600 ~/.git-credentials
git remote set-url origin "https://github.com/$GITHUB_REPO.git"   # strip any token from the URL
git config --global user.email georgemadas5@gmail.com
git config --global user.name GeorgeMadas
git config --global --add safe.directory /workspace/qwen-image
```

**Check:** `git -C /workspace/qwen-image fetch && git -C /workspace/qwen-image status -sb` shows
`## main...origin/main` with no error.

## 3. Python packages

```bash
cd /workspace/qwen-image
uv pip install --python /venv/main/bin/python -r deploy/requirements.txt
```

`diffusers` is pinned to a git commit because `QwenImage21Pipeline` isn't in a release; keep the pin.

**Check:**
```bash
/venv/main/bin/python -c "from diffusers import QwenImage21Pipeline; import spandrel, gguf, peft, fastapi, uvicorn; print('ok')"
```

## 4. Service + portal entry

Install the supervisor service exactly as on the reference instance:

```bash
cp /workspace/qwen-image/deploy/service/qwen-image.sh /opt/supervisor-scripts/qwen-image.sh
chmod +x /opt/supervisor-scripts/qwen-image.sh
cp /workspace/qwen-image/deploy/service/qwen-image.conf /etc/supervisor/conf.d/qwen-image.conf
```

Add the portal entry (use the port chosen in step 1 as `external_port`; `internal_port` stays
`17860`, which is where `app.py` listens):

```bash
EXT=10100
python3 -c "import yaml; p='/etc/portal.yaml'; d=yaml.safe_load(open(p)) or {'applications':{}}; \
d['applications']['Qwen Image']={'hostname':'localhost','external_port':$EXT,'internal_port':17860,'open_path':'/','name':'Qwen Image'}; \
yaml.safe_dump(d, open(p,'w'), sort_keys=False)"
supervisorctl restart caddy
```

**Do not start the service yet** — without the models from step 5 it would start a 37 GB download
in the foreground of the app. (`supervisorctl reread && supervisorctl update` is done in step 8.)

**Check:** `grep -A5 'Qwen Image' /etc/portal.yaml` shows the entry; `supervisorctl status caddy` is RUNNING.

## 5. Models

Upscalers (into the app dir; `app.py` verifies the SHA-256 of both):

```bash
cd /workspace/qwen-image/upscalers
wget -q -O 4x-UltraSharp.safetensors https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.safetensors
wget -q -O RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
sha256sum *
#   36a340b5509b699d2c06cb445ddc1d3d39199ac734d889ed6d7915f60e05bcbc  4x-UltraSharp.safetensors
#   4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1  RealESRGAN_x4plus.pth
```

Diffusion weights into the Hugging Face cache (long download — run it in the background and
monitor `/workspace/prefetch.log`):

```bash
cd /workspace/qwen-image
nohup /venv/main/bin/python deploy/prefetch_models.py > /workspace/prefetch.log 2>&1 &
```

What it downloads into `$HF_HOME` (`/workspace/.hf_home`):
- `KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF` → `qwen-image-2.1-UC-BF16.gguf` ("Full")
  and `qwen-image-2.1-UC-Q4_K_M.gguf` ("Small")
- `Qwen/Qwen-Image-2.1` @ `790c92633540aa0cb11d9abf19eb46d861714758` — text encoder, VAE, processor,
  scheduler, configs (transformer weights skipped; the GGUF replaces them)

- `Qwen/Qwen-Image-2.1-PE-T2I` @ `f3ed7985` and `Qwen/Qwen-Image-2.1-PE-I2I` @ `72927bc0` — the official
  prompt rewriters (Qwen3.5-VL 9B, ~19 GB each). `prompt_rewriter.py` loads them lazily on the first
  rewrite; only one sits on the GPU at a time (~20 GB), the other waits in CPU RAM.

Style LoRAs (WarmBloodAban, prithivMLmods, SimpleTuner, AIImageStudio, e-n-v-y, Viggle, …) are
**not** prefetched — `app.py` downloads each one the first time it's picked in the UI.

**Check:** `/workspace/prefetch.log` ends with `Model prefetch complete`, and both `sha256sum`
lines match the values above.

## 6. App data that doesn't come from git

These are per-instance and intentionally not in the repo — nothing to do, just know they exist:
`outputs/` (generated images), `history.jsonl` (Library history), `loras/*` (user-added local LoRAs).
`app.py` creates them as needed. `prompts.json` and `settings.json` *are* in git.

## 7. Claude Code extras

The user installs Claude Code itself. You set up its config:

```bash
mkdir -p ~/.claude/skills ~/.claude/projects/-workspace/memory
cd /workspace/qwen-image/deploy/claude
cp -n settings.json ~/.claude/settings.json
cp -rn skills/* ~/.claude/skills/
cp -n memory/*.md ~/.claude/projects/-workspace/memory/
```

- `settings.json` — `{"theme": "dark"}`
- skills (13): brandkit, design-taste-frontend, design-taste-frontend-v1, gpt-taste,
  high-end-visual-design, image-to-code, imagegen-frontend-mobile, imagegen-frontend-web,
  industrial-brutalist-ui, minimalist-ui, qwen-image-2-1-prompter, redesign-existing-projects,
  stitch-design-taste
- memories: `no-test-generations` (the "never generate to test" rule above) and `obscura-repo-setup`
  (this repo, SETUP.md, commit + push after app changes). The memory path is
  keyed to the working directory `/workspace`, so Claude should be started from `/workspace`.

`cp -n` never overwrites; if a file already exists, diff it and ask the user before replacing.

**Check:** `ls ~/.claude/skills | wc -l` ≥ 13 and `cat ~/.claude/projects/-workspace/memory/MEMORY.md`
lists both memories. Tell the user skills/memory load on the **next** Claude Code session.

## 8. Start and verify

```bash
supervisorctl reread && supervisorctl update   # registers + starts qwen-image
supervisorctl status qwen-image
tail -n 40 /var/log/portal/qwen-image.log
```

Model load takes a few minutes (BF16 GGUF is dequantized into GPU memory). The log prints
`Loading checkpoint …`, `Loaded … tensors`, `… successfully initialized on CUDA`; uvicorn runs at
`log_level="warning"`, so there is no "server started" line — poll the port instead:

```bash
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:17860/)" = 200 ]; do sleep 10; done; echo up
vast-capabilities | jq '.services[] | select(.name=="Qwen Image")'       # gives the public direct_url
```

**Check:** status `RUNNING`, local request returns `200`, `nvidia-smi` shows the python process
holding GPU memory. **Do not** generate an image to test.

Finish by giving the user the `direct_url` (it needs their `OPEN_BUTTON_TOKEN`, as on the old
instance) and telling them it's ready to test.

---

## 9. ComfyUI (video workflows)

Obscura's **Workflows** tab runs ComfyUI graphs: the built-in video presets (Image → Video 14B, its 4-step
fast version, Motion transfer (the user's DREAMOFM ultimate motion workflow: Wan 2.2 Animate with
pose detection), and the 5B draft model) plus any workflow the user imports.
`deploy/provision.sh` does this step automatically on new instances; by hand it is one script:

```bash
bash /workspace/qwen-image/deploy/install_comfyui.sh   # idempotent; ~80 GB of models
```

What it sets up:
- `/workspace/ComfyUI` at a pinned commit, with ComfyUI-Manager, VideoHelperSuite, KJNodes, and the packs the
  user's *DREAMOFM ultimate motion* workflow needs: WanVideoWrapper, WanAnimatePreprocess, teskors-utils,
  rgthree-comfy (plus WanAnimatePlus and Dynamic-Lora-Scheduler). Commits are pinned in the script.
  Workflows saved in the editor live in `/workspace/ComfyUI/user/default/workflows/` (not in git).
- **Its own venv `/venv/comfy`** with the same `torch 2.11.0+cu128` as `/venv/main`. ComfyUI-Manager installs
  whatever a node asks for; in its own venv that can't break Obscura's pinned `transformers`/`diffusers`.
- Service `comfyui` (`deploy/service/comfyui.sh` + `.conf`), listening on `127.0.0.1:18188`, and the portal
  entry `ComfyUI`: external `10200` → `18188` behind Caddy token auth.
- Models from `deploy/prefetch_comfy.py` into `/workspace/ComfyUI/models/{diffusion_models,loras,text_encoders,vae,clip_vision}`.
- Presets live in `workflows/` (API graph + `.meta.json` form). Regenerate them with
  `python deploy/build_presets.py` (Motion transfer's graph comes from `deploy/presets/`); workflows the user imports are saved there too and synced through git.

How the GPU is shared: Qwen keeps ~60–88 GB of VRAM while idle, so a ComfyUI job first moves Qwen (and the
prompt rewriter) to CPU RAM, runs, then asks ComfyUI to free its VRAM (`POST /free`). The next image render
moves Qwen back, which takes several seconds ("Loading Qwen back to GPU"). Jobs share Obscura's single queue, so
a video never overlaps a render. Runs started in the ComfyUI **editor** don't go through that queue: use
**Free GPU for the editor** on the Workflows tab first.

**Check** (never run a workflow to test):
```bash
supervisorctl status comfyui                                   # RUNNING
curl -s 127.0.0.1:18188/system_stats | head -c 200             # JSON
curl -s 127.0.0.1:17860/api/workflows | python3 -c "import json,sys; print([w['slug'] for w in json.load(sys.stdin)['items']])"
vast-capabilities | jq '.services[] | select(.name=="ComfyUI") | .direct_url'
```

## Keeping instances in sync

Both instances are clones of `GeorgeMadas/obscura`. After changing app code: commit, `git push`;
on the other instance `git pull` then `supervisorctl restart qwen-image`. If `deploy/requirements.txt`
changed, rerun step 3 before restarting. If ComfyUI files changed (`deploy/install_comfyui.sh`,
`deploy/service/comfyui.*`, `deploy/prefetch_comfy.py`), rerun step 9. Never commit `outputs/` or `history.jsonl` (gitignored).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no kernel image is available for execution on the device` | torch too old for the GPU → step 1 reinstall with cu128 |
| Service exits instantly, log says it was skipped | Portal entry name isn't exactly `Qwen Image` → step 4 |
| Portal link 502 | App still loading the model; watch the log |
| `Checksum verification failed` | Corrupt GGUF download → delete that blob under `$HF_HOME/hub/models--KasugaiSakura--*/blobs/` and rerun prefetch |
| Rewrite popup says the GPU ran out of memory | A 4MP render + the ~20 GB rewriter didn't fit at once → wait for the render, or turn the rewrite toggle (wand icon) off |
| CUDA OOM at load | GPU < ~48 GB → user switches to "Small" in the UI (writes `settings.json`) |
| Workflows tab says "ComfyUI isn't running" | `supervisorctl start comfyui`; log in `/var/log/portal/comfyui.log` |
| ComfyUI service skipped at start | Portal entry name isn't exactly `ComfyUI` → rerun `deploy/install_comfyui.sh` |
| A video job fails with out of memory | A run from the ComfyUI editor or the prompt rewriter was holding VRAM → click **Free GPU for the editor**, wait, retry |
| "Not enough free GPU memory to bring Qwen back" | ComfyUI still holds models (e.g. **Keep video models loaded** is on, or an editor run) → turn it off / wait, then render again |
| Imported workflow rejected as "editor format" | In ComfyUI use **Workflow → Export (API)**, not the normal save |
| `git push` asks for a password | Step 2 credentials missing or token revoked — ask the user for a new token, update `deploy/github.env` |
