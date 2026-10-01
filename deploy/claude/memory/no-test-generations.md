---
name: no-test-generations
description: Never trigger image generations or upscales to test the Qwen/Obscura app; the user does all testing
metadata:
  node_type: memory
  type: feedback
  originSessionId: 623e2e29-e5f8-45ac-9e77-be44a25e5684
  modified: 2026-09-27T15:06:56.536Z
---

Do not generate images (renders, batches, upscales) yourself when working on the Obscura / Qwen Image app in /workspace/qwen-image. The user does the testing.

**Why:** The user stopped a browser test and said "from now on you will not generate images yourself OK I will do the testing" (2026-09-27). Earlier test renders cluttered their Library and used their GPU.

**How to apply:** Verify changes with syntax checks (`node --check`, `python -m py_compile`) and code reading only, then tell the user what to test. Do not call /api/generate or /api/upscale, and do not drive the UI to click Generate or Upscale. Ask before any other testing that touches their images.
