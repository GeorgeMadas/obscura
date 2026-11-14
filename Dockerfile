# Obscura runtime image for Vast.ai.
#
# Bakes in the slow, rarely-changing parts so a new instance skips ~20 min of installs:
#   - Obscura's Python packages and the prompt-rewriter kernels in /venv/main
#   - ComfyUI + the pinned custom node packs in /opt/ComfyUI, with its own venv /venv/comfy
#   - Claude Code
#
# Deliberately NOT in the image (see SETUP.md "Custom Docker image"):
#   - the app code: deploy/provision.sh clones / fast-forwards GeorgeMadas/obscura into /workspace/qwen-image
#     at boot, so app changes need only a git push
#   - the ~155 GB of model weights: they live on /workspace (attach a persistent Vast volume there)
#
# Built and pushed to ghcr.io/georgemadas/obscura by .github/workflows/docker-image.yml.
FROM vastai/pytorch:cuda-12.8.1-auto

ENV HF_HOME=/workspace/.hf_home \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /opt/obscura

# 1. Claude Code (changes least often, so it sits in the lowest layer)
RUN curl -fsSL https://claude.ai/install.sh | bash \
 && ln -sf /root/.local/bin/claude /usr/local/bin/claude

# 2. Prompt-rewriter kernels. causal-conv1d compiles against the base image's torch; keep MAX_JOBS low so the
#    CI runner doesn't run out of RAM. If the build fails, provision.sh installs it at boot instead.
RUN uv pip install --python /venv/main/bin/python ninja==1.13.2 flash-linear-attention==0.5.2 \
 && (MAX_JOBS=2 uv pip install --python /venv/main/bin/python --no-build-isolation causal-conv1d==1.7.0 \
     || echo "causal-conv1d build failed in the image; provision.sh will install it at boot")

# 3. ComfyUI at the pinned commit + node packs + /venv/comfy (same script provision.sh runs at boot)
COPY deploy/install_comfyui.sh /opt/obscura/install_comfyui.sh
# Models and per-instance data are symlinked to /workspace (the persistent volume), so they stay out of the image
RUN COMFY_DIR=/opt/ComfyUI COMFY_BUILD_ONLY=1 bash /opt/obscura/install_comfyui.sh \
 && rm -rf /opt/ComfyUI/models /opt/ComfyUI/input /opt/ComfyUI/output /opt/ComfyUI/user \
 && ln -s /workspace/models/comfyui /opt/ComfyUI/models \
 && ln -s /workspace/comfyui-data/input /opt/ComfyUI/input \
 && ln -s /workspace/comfyui-data/output /opt/ComfyUI/output \
 && ln -s /workspace/comfyui-data/user /opt/ComfyUI/user \
 && rm -rf /root/.cache/uv /root/.cache/pip

# 4. Obscura's pinned Python packages (changes most often, so it's the top layer)
COPY deploy/requirements.txt /opt/obscura/requirements.txt
RUN uv pip install --python /venv/main/bin/python -r /opt/obscura/requirements.txt \
 && /venv/main/bin/python -c "import torch, diffusers, fastapi, fla; print('torch', torch.__version__)" \
 && rm -rf /root/.cache/uv /root/.cache/pip

LABEL org.opencontainers.image.source=https://github.com/GeorgeMadas/obscura \
      org.opencontainers.image.description="Obscura (Qwen Image + ComfyUI) runtime for Vast.ai"
