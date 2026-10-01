# Vast.ai template: Qwen Image (Obscura)

Create it at https://cloud.vast.ai/templates/ → **New Template** (or edit a copy of template #740692).

| Field | Value |
|---|---|
| Image | `vastai/pytorch:cuda-12.8.1-auto` |
| Launch mode | Jupyter + SSH (same as the original instance) |
| Disk | 100 GB (models ~32 GB + outputs) |
| Visibility | **Private** (env holds your GitHub token) |

**Ports** (`-p` entries): `1111`, `6006`, `8080`, `8384`, `10100`, `10200`, `72299`

**Environment variables**

```
OPEN_BUTTON_PORT=1111
OPEN_BUTTON_TOKEN=1
JUPYTER_DIR=/
DATA_DIRECTORY=/workspace/
PORTAL_CONFIG=localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:10100:17860:/:Qwen Image
QWEN_REPO=<github-user>/<repo>
GITHUB_TOKEN=<fine-grained PAT, Contents: read & write on that repo>
```

Optional: `HF_TOKEN=<token>` for faster / rate-limit-free model downloads.

**On-start script** (replaces the default `entrypoint.sh`):

```bash
curl -fsSL -H "Authorization: token $GITHUB_TOKEN" \
  "https://raw.githubusercontent.com/$QWEN_REPO/main/deploy/provisioning.yaml" \
  -o /provisioning.yaml || echo "Could not fetch provisioning manifest"
entrypoint.sh
```

## What happens on first boot

1. The manifest above is downloaded; the base image's provisioner runs it
   (progress: `/var/log/portal/provisioning.log` or the Instance Portal).
2. This repo is cloned to `/workspace/qwen-image` and `deploy/requirements.txt` installed.
3. Upscaler weights download into `upscalers/`; the `qwen-image` service is registered.
4. `deploy/prefetch_models.py` pulls the GGUF checkpoint + Qwen-Image-2.1 companion into `$HF_HOME`.
5. Claude Code is installed with `deploy/claude/` settings, memory and skills
   (run `claude` and log in once per instance).
6. Provisioning finishes and the app starts on the **Qwen Image** portal link (port 10100).

Not carried over: `outputs/`, `history.jsonl`, `loras/` contents — they're per-instance.

## Keeping instances in sync

The app directory is a git clone on every instance. `git pull` / `git push` work
using `GITHUB_TOKEN` (via the credential helper in `/etc/gitconfig`). Then
`supervisorctl restart qwen-image` to load changes.
