"""Official Qwen Image 2.1 prompt rewriters (PE-T2I and PE-I2I), run before a render so the person can review the result.

Both are fine-tuned Qwen3.5-VL 9B checkpoints that think first, then answer with JSON:
  {"rewritten_prompt": "...", "wh_ratio": "3:2"}                       (T2I)
  {"rewritten_prompt": "...", "wh_ratio": "", "ratio_follow": "<image1>"}  (I2I)
Only one sits on the GPU at a time; the other waits in CPU memory so switching modes takes seconds, not a reload.
"""
import json
import re
import threading
import time
import traceback
import uuid

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

REWRITERS = {
    "t2i": {"repo": "Qwen/Qwen-Image-2.1-PE-T2I", "revision": "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b", "max_new_tokens": 16256},
    "i2i": {"repo": "Qwen/Qwen-Image-2.1-PE-I2I", "revision": "72927bc08afc99b7888ceb7d7d51a12db3700bbd", "max_new_tokens": 24000},
}
# The rewriter only needs to understand the references, not see every pixel
REF_MAX_SIDE = 1024
MAX_REFS = 10


class RewriteError(Exception):
    """A problem the person can act on; shown verbatim in the UI."""


class _Cancel(StoppingCriteria):
    def __init__(self, job):
        self.job = job

    def __call__(self, input_ids, scores, **kwargs):
        return self.job["cancel"]


_LOADED: dict[str, tuple] = {}   # kind -> (model, processor, system_prompt)
_ON_GPU: list[str] = []          # kind currently on the GPU, if any
_LOCK = threading.Lock()         # one rewrite at a time
JOBS: dict[str, dict] = {}


def _ensure(kind: str, job: dict):
    spec = REWRITERS[kind]
    if kind not in _LOADED:
        job["phase"] = "Loading rewriter"
        processor = AutoProcessor.from_pretrained(spec["repo"], revision=spec["revision"])
        model = AutoModelForImageTextToText.from_pretrained(spec["repo"], revision=spec["revision"], dtype=torch.bfloat16).eval()
        system_prompt = open(hf_hub_download(spec["repo"], "system_prompt.txt", revision=spec["revision"]), encoding="utf-8").read().strip()
        _LOADED[kind] = (model, processor, system_prompt)
    model = _LOADED[kind][0]
    if _ON_GPU != [kind]:
        job["phase"] = "Moving rewriter to GPU"
        for other in _ON_GPU:
            _LOADED[other][0].to("cpu")
        _ON_GPU.clear()
        torch.cuda.empty_cache()
        model.to("cuda")
        _ON_GPU.append(kind)
    return _LOADED[kind]


def _prepare(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        img = white
    img = img.convert("RGB")
    img.thumbnail((REF_MAX_SIDE, REF_MAX_SIDE), Image.LANCZOS)
    return img


def _parse(text: str) -> dict:
    thinking, sep, answer = text.partition("</think>")
    if not sep:
        answer = thinking
    answer = answer.strip()
    answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer)
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", answer, re.S)
        if not m:
            raise RewriteError("The rewriter did not return a prompt. Try again.")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            raise RewriteError("The rewriter's answer was malformed. Try again.")
    prompt = str(data.get("rewritten_prompt") or "").strip()
    if not prompt:
        raise RewriteError("The rewriter returned an empty prompt. Try again.")
    return {
        "prompt": prompt,
        "wh_ratio": str(data.get("wh_ratio") or "").strip(),
        "ratio_follow": str(data.get("ratio_follow") or "").strip(),
    }


def _run(job: dict):
    kind = job["kind"]
    try:
        with _LOCK:
            if job["cancel"]:
                job["status"] = "cancelled"
                return
            job["status"] = "running"
            model, processor, system_prompt = _ensure(kind, job)
            images = job.pop("images", [])
            content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": job["input"]}]
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": content},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=True,
            ).to(model.device)
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            failure = []

            def generate():
                try:
                    with torch.inference_mode():
                        model.generate(
                            **inputs, streamer=streamer, max_new_tokens=REWRITERS[kind]["max_new_tokens"],
                            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                            stopping_criteria=StoppingCriteriaList([_Cancel(job)]),
                        )
                except Exception as e:  # surfaced after the stream ends
                    failure.append(e)
                    streamer.end()

            job["phase"] = "Thinking"
            job["started"] = time.time()
            worker = threading.Thread(target=generate, daemon=True, name="prompt-rewrite")
            worker.start()
            text = ""
            for piece in streamer:
                text += piece
                job["tokens"] += 1
                if "</think>" in text:
                    job["phase"] = "Writing prompt"
                job["thinking"] = text.partition("</think>")[0].replace("<think>", "").strip()[-600:]
            worker.join()
            del inputs
            if job["cancel"]:
                job["status"] = "cancelled"
                return
            if failure:
                raise failure[0]
            job["result"] = _parse(text)
            job["status"] = "done"
    except RewriteError as e:
        job.update(status="error", error=str(e))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        job.update(status="error", error="The GPU ran out of memory while rewriting. Try again once the current render finishes.")
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=f"Prompt rewrite failed: {e}")
    finally:
        job.pop("images", None)
        job["finished"] = time.time()
        _prune()


def park():
    """Move the rewriter on the GPU (if any) to CPU memory, so ComfyUI gets the VRAM. Waits for a running rewrite."""
    with _LOCK:
        for kind in _ON_GPU:
            _LOADED[kind][0].to("cpu")
        _ON_GPU.clear()
        torch.cuda.empty_cache()


def _prune():
    cutoff = time.time() - 1800
    for jid in [k for k, j in JOBS.items() if j.get("finished", time.time()) < cutoff]:
        JOBS.pop(jid, None)


def start(prompt: str, refs: list[Image.Image]) -> dict:
    """Queue a rewrite. With references it uses PE-I2I, otherwise PE-T2I."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise RewriteError("Describe what you want to see first.")
    refs = refs[:MAX_REFS]
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "kind": "i2i" if refs else "t2i", "status": "queued", "phase": "Queued",
        "input": prompt, "images": [_prepare(r) for r in refs], "tokens": 0, "thinking": "",
        "cancel": False, "created": time.time(), "result": None, "error": None,
    }
    threading.Thread(target=_run, args=(JOBS[job_id],), daemon=True, name="prompt-rewrite-job").start()
    return view(JOBS[job_id])


def view(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "kind", "status", "phase", "tokens", "thinking", "result", "error", "created", "started")} | {"server_time": time.time()}


def get(job_id: str) -> dict | None:
    job = JOBS.get(job_id)
    return view(job) if job else None


def cancel(job_id: str) -> dict | None:
    job = JOBS.get(job_id)
    if not job:
        return None
    job["cancel"] = True
    if job["status"] == "queued":
        job["status"] = "cancelled"
    return view(job)
