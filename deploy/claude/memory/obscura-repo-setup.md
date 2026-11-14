---
name: obscura-repo-setup
description: The Obscura/Qwen Image app is in private GitHub repo GeorgeMadas/obscura; SETUP.md there rebuilds a new Vast instance
metadata:
  type: project
---

/workspace/qwen-image is a git clone of private repo GeorgeMadas/obscura (created 2026-10-01). Its SETUP.md is a step-by-step guide written for Claude to rebuild the whole instance (packages, service, portal entry, models, Claude skills/memory, git credentials) on a fresh Vast.ai instance for parallel work. The user installs Claude Code themselves, then asks Claude to follow SETUP.md.

**Why:** The user wants a second instance without Vast's copy feature; they dropped the Vast template approach.

**How to apply:** When asked to set up a new instance, follow SETUP.md step by step. After app changes, commit and push so the other instance can pull. The GitHub token comes from GITHUB_TOKEN in the Vast account env vars (no longer kept in deploy/github.env). New instances can use the prebuilt image ghcr.io/georgemadas/obscura (see SETUP.md "Custom Docker image"). Related: [[no-test-generations]]
