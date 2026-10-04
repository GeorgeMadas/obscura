"""Bridge between Obscura and a local ComfyUI server: a workflow store, form fields, and running jobs.

Workflows are ComfyUI graphs in API format ("Export (API)" in the editor), stored as
workflows/<slug>.json with a sidecar workflows/<slug>.meta.json describing the form:

  {"name": ..., "description": ..., "output": "video" | "image", "preset": bool,
   "size": {"width": <field key>, "height": <field key>, "from": <image field key>, "multiple": 16},
   "fields": [{"key", "label", "type", "default", "targets": [[node_id, input_name, transform?], ...],
               "min", "max", "step", "hidden", "help"}]}

Field types: text, int, float, seed, image, video, bool. The only transform is "half" (value // 2), used to
split steps between Wan 2.2's high-noise and low-noise samplers.
"""
import copy
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
import websocket

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:18188").rstrip("/")
WORKFLOW_DIR = Path(os.environ.get("OBSCURA_WORKFLOW_DIR", Path(__file__).resolve().parent / "workflows"))
WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
STORE_LOCK = threading.Lock()
MAX_SEED = 2**31 - 1

FIELD_TYPES = {"text", "int", "float", "seed", "image", "video", "bool"}
SLUG = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")
SAMPLERS = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", "WanVideoSampler")
# Inputs exposed automatically when a workflow is imported: input name -> (type, label)
NUMBER_INPUTS = {
    "steps": ("int", "Steps"), "cfg": ("float", "CFG"), "denoise": ("float", "Denoise"),
    "width": ("int", "Width"), "height": ("int", "Height"), "length": ("int", "Frames"),
    "fps": ("float", "FPS"), "frame_rate": ("float", "FPS"),
}
TEXT_INPUTS = ("text", "prompt", "positive_prompt", "negative_prompt", "text_g", "text_l")
IMAGE_LOADERS = {"LoadImage": "image"}
VIDEO_LOADERS = {"LoadVideo": "file", "VHS_LoadVideo": "video", "VHS_LoadVideoPath": "video"}
VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".gif", ".avi", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}


class BridgeError(Exception):
    """A problem the person can act on; app.py turns it into a UserError."""


class Interrupted(Exception):
    pass


# ============================================================
# ComfyUI server
# ============================================================
def _get(path: str, timeout: float = 5, **kw):
    try:
        r = requests.get(COMFY_URL + path, timeout=timeout, **kw)
    except requests.RequestException:
        raise BridgeError("ComfyUI isn't running. Start it with `supervisorctl start comfyui`.")
    r.raise_for_status()
    return r


def _post(path: str, timeout: float = 10, **kw):
    try:
        return requests.post(COMFY_URL + path, timeout=timeout, **kw)
    except requests.RequestException:
        raise BridgeError("ComfyUI isn't running. Start it with `supervisorctl start comfyui`.")


def status() -> dict:
    """Up/down, VRAM and queue length, without raising."""
    try:
        stats = requests.get(COMFY_URL + "/system_stats", timeout=2).json()
        q = requests.get(COMFY_URL + "/queue", timeout=2).json()
    except (requests.RequestException, ValueError):
        return {"up": False}
    dev = (stats.get("devices") or [{}])[0]
    return {
        "up": True,
        "version": stats.get("system", {}).get("comfyui_version"),
        "vram_total": dev.get("vram_total"),
        "vram_free": dev.get("vram_free"),
        "running": len(q.get("queue_running", [])),
        "pending": len(q.get("queue_pending", [])),
    }


def free(unload_models: bool = True):
    """Ask ComfyUI to drop its models and cached VRAM. Its worker applies this between prompts."""
    try:
        requests.post(COMFY_URL + "/free", json={"unload_models": unload_models, "free_memory": True}, timeout=3)
    except requests.RequestException:
        pass


def busy() -> bool:
    """True while ComfyUI has prompts running or queued (from Obscura or the editor)."""
    s = status()
    return bool(s.get("up") and (s.get("running") or s.get("pending")))


def object_info() -> dict:
    return _get("/object_info", timeout=30).json()


# ============================================================
# Workflow store
# ============================================================
def is_api_format(graph) -> bool:
    return (isinstance(graph, dict) and bool(graph)
            and all(isinstance(n, dict) and "class_type" in n and isinstance(n.get("inputs"), dict) for n in graph.values()))


def check_graph(graph) -> dict:
    if isinstance(graph, str):
        try:
            graph = json.loads(graph)
        except json.JSONDecodeError as e:
            raise BridgeError(f"That isn't valid JSON: {e}")
    if isinstance(graph, dict) and "nodes" in graph and "links" in graph:
        raise BridgeError("This is the editor format. In ComfyUI use Workflow → Export (API) and import that file.")
    if isinstance(graph, dict) and is_api_format(graph.get("prompt")):
        graph = graph["prompt"]
    if not is_api_format(graph):
        raise BridgeError("That doesn't look like a ComfyUI API workflow. In ComfyUI use Workflow → Export (API).")
    return graph


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "workflow"
    base, n = s, 2
    while (WORKFLOW_DIR / f"{s}.json").exists():
        s = f"{base}-{n}"
        n += 1
    return s


def _paths(slug: str) -> tuple[Path, Path]:
    if not SLUG.match(slug or ""):
        raise BridgeError("Unknown workflow.")
    return WORKFLOW_DIR / f"{slug}.json", WORKFLOW_DIR / f"{slug}.meta.json"


def load(slug: str) -> tuple[dict, dict]:
    graph_path, meta_path = _paths(slug)
    if not graph_path.is_file():
        raise BridgeError("That workflow no longer exists.")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta.setdefault("name", slug)
    meta.setdefault("fields", detect_fields(graph))
    meta.setdefault("output", guess_output(graph))
    return graph, meta


def _write(slug: str, graph: dict | None, meta: dict):
    graph_path, meta_path = _paths(slug)
    for path, data in ((graph_path, graph), (meta_path, meta)):
        if data is None:
            continue
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def public(slug: str, meta: dict) -> dict:
    return {
        "slug": slug,
        "name": meta.get("name", slug),
        "description": meta.get("description", ""),
        "output": meta.get("output", "image"),
        "preset": bool(meta.get("preset")),
        "size": meta.get("size"),
        "fields": meta.get("fields", []),
        "models": meta.get("models", []),
        "order": meta.get("order", 100),
    }


def list_workflows() -> list[dict]:
    items = []
    for path in WORKFLOW_DIR.glob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        try:
            _, meta = load(path.stem)
        except (BridgeError, json.JSONDecodeError, OSError):
            continue
        items.append(public(path.stem, meta))
    # Presets first in their own order, then imports newest first
    return sorted(items, key=lambda w: (not w["preset"], w["order"], -_mtime(w["slug"])))


def _mtime(slug: str) -> float:
    try:
        return (WORKFLOW_DIR / f"{slug}.json").stat().st_mtime
    except OSError:
        return 0


def import_workflow(name: str, graph, description: str = "") -> dict:
    graph = check_graph(graph)
    name = (name or "").strip()[:80] or "Imported workflow"
    meta = {"name": name, "description": (description or "").strip()[:400], "output": guess_output(graph),
            "fields": detect_fields(graph), "imported": time.time()}
    with STORE_LOCK:
        slug = _slugify(name)
        _write(slug, graph, meta)
    return public(slug, meta)


def update_workflow(slug: str, body: dict) -> dict:
    with STORE_LOCK:
        graph, meta = load(slug)
        if "name" in body:
            meta["name"] = str(body["name"] or "").strip()[:80] or meta["name"]
        if "description" in body:
            meta["description"] = str(body["description"] or "").strip()[:400]
        if body.get("output") in ("image", "video"):
            meta["output"] = body["output"]
        if "fields" in body:
            meta["fields"] = clean_fields(graph, body["fields"])
        _write(slug, None, meta)
    return public(slug, meta)


def delete_workflow(slug: str):
    graph_path, meta_path = _paths(slug)
    with STORE_LOCK:
        _, meta = load(slug)
        if meta.get("preset"):
            raise BridgeError("Built-in presets can't be deleted.")
        graph_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)


def clean_fields(graph: dict, fields) -> list[dict]:
    """Validate a field list from the Edit fields dialog against the graph."""
    if not isinstance(fields, list):
        raise BridgeError("Fields must be a list.")
    out, keys = [], set()
    for f in fields[:60]:
        if not isinstance(f, dict):
            continue
        ftype = f.get("type")
        if ftype not in FIELD_TYPES:
            raise BridgeError(f"Unknown field type: {ftype}")
        targets = []
        for t in f.get("targets") or []:
            if not isinstance(t, list) or len(t) < 2:
                continue
            node, inp = str(t[0]), str(t[1])
            if node not in graph:
                raise BridgeError(f"Field {f.get('label')!r} points at node {node}, which isn't in the workflow.")
            if inp not in graph[node]["inputs"]:
                raise BridgeError(f"Node {node} has no input called {inp!r}.")
            targets.append([node, inp] + ([t[2]] if len(t) > 2 and t[2] == "half" else []))
        if not targets:
            raise BridgeError(f"Field {f.get('label')!r} needs at least one node input.")
        key = re.sub(r"[^\w]", "_", str(f.get("key") or f"f{len(out)}"))[:40]
        while key in keys:
            key += "_"
        keys.add(key)
        clean = {"key": key, "label": str(f.get("label") or key)[:60], "type": ftype, "targets": targets,
                 "default": f.get("default"), "hidden": bool(f.get("hidden"))}
        for k in ("min", "max", "step"):
            if isinstance(f.get(k), (int, float)):
                clean[k] = f[k]
        if f.get("help"):
            clean["help"] = str(f["help"])[:200]
        out.append(clean)
    return out


# ============================================================
# Input detection
# ============================================================
def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)


def _title(graph: dict, node_id: str) -> str:
    n = graph[node_id]
    return (n.get("_meta") or {}).get("title") or n["class_type"]


def _consumers(graph: dict, node_id: str) -> list[str]:
    """Input names that read from this node anywhere in the graph."""
    return [name for n in graph.values() for name, v in n["inputs"].items() if _is_link(v) and v[0] == node_id]


def _node_order(graph: dict) -> list[str]:
    return sorted(graph, key=lambda k: (int(k) if k.isdigit() else 10**9, k))


def guess_output(graph: dict) -> str:
    for n in graph.values():
        ct = n["class_type"]
        if ct in ("SaveVideo", "SaveWEBM", "SaveAnimatedWEBP", "SaveAnimatedPNG") or ct.startswith("VHS_VideoCombine"):
            return "video"
    return "image"


def detect_fields(graph: dict) -> list[dict]:
    fields: list[dict] = []
    grouped: dict[tuple, dict] = {}
    prompts = []
    for nid in _node_order(graph):
        node = graph[nid]
        ct, inputs = node["class_type"], node["inputs"]
        if ct in IMAGE_LOADERS and isinstance(inputs.get(IMAGE_LOADERS[ct]), str):
            n_img = sum(1 for f in fields if f["type"] == "image") + 1
            fields.append({"key": f"image_{nid}", "label": "Image" if n_img == 1 else f"Image {n_img}", "type": "image",
                           "default": "", "targets": [[nid, IMAGE_LOADERS[ct]]]})
            continue
        if ct in VIDEO_LOADERS and isinstance(inputs.get(VIDEO_LOADERS[ct]), str):
            fields.append({"key": f"video_{nid}", "label": "Video", "type": "video", "default": "",
                           "targets": [[nid, VIDEO_LOADERS[ct]]]})
            continue
        for name, value in inputs.items():
            if _is_link(value):
                continue
            if name in TEXT_INPUTS and isinstance(value, str):
                title = _title(graph, nid)
                negative = "negative" in title.lower() or "negative" in name or "negative" in _consumers(graph, nid)
                prompts.append((negative, nid, name, value, title))
            elif name in ("seed", "noise_seed") and isinstance(value, int):
                _group(grouped, fields, ("seed",), {"key": "seed", "label": "Seed", "type": "seed", "default": value,
                                                    "min": 0, "max": MAX_SEED}, nid, name)
            elif name in NUMBER_INPUTS and isinstance(value, (int, float)) and not isinstance(value, bool):
                ftype, label = NUMBER_INPUTS[name]
                _group(grouped, fields, (label, value), {"key": f"{name}_{nid}", "label": label, "type": ftype,
                                                         "default": value}, nid, name)
    # Prompts lead the form: positive first, then negative
    lead = []
    n_pos = sum(1 for p in prompts if not p[0])
    for negative, nid, name, value, title in sorted(prompts, key=lambda p: p[0]):
        label = "Negative prompt" if negative else "Prompt" if n_pos == 1 else f"Prompt ({title})"
        lead.append({"key": f"{'neg' if negative else 'prompt'}_{nid}", "label": label, "type": "text",
                     "default": value, "targets": [[nid, name]], "hidden": False})
    return lead + fields


def _group(grouped: dict, fields: list, key: tuple, field: dict, nid: str, name: str):
    """Equal values of the same kind across nodes (both samplers' seed, say) become one field."""
    if key in grouped:
        grouped[key]["targets"].append([nid, name])
        return
    field["targets"] = [[nid, name]]
    grouped[key] = field
    fields.append(field)


# ============================================================
# Running a workflow
# ============================================================
def _cast(field: dict, value):
    t = field["type"]
    try:
        if t in ("int", "seed"):
            v = int(float(value))
        elif t == "float":
            v = float(value)
        elif t == "bool":
            v = value in (True, "true", "1", 1, "on")
        else:
            return str(value if value is not None else "")
    except (TypeError, ValueError):
        raise BridgeError(f"{field['label']} must be a number.")
    if t in ("int", "float", "seed"):
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and v < lo:
            raise BridgeError(f"{field['label']} must be at least {lo}.")
        if hi is not None and v > hi:
            raise BridgeError(f"{field['label']} must be at most {hi}.")
    return v


def resolve_values(meta: dict, values: dict, randomize: dict) -> dict:
    """Form values by field key, cast and checked; seeds are drawn here when randomized."""
    out = {}
    for f in meta.get("fields", []):
        key = f["key"]
        if f["type"] in ("image", "video"):
            continue
        raw = values.get(key, f.get("default"))
        if f["type"] == "seed" and (randomize.get(key, True) or raw in (None, "")):
            out[key] = random.randint(0, MAX_SEED)
        elif raw is None:
            continue
        else:
            out[key] = _cast(f, raw)
    return out


def patch(graph: dict, meta: dict, values: dict, media: dict) -> dict:
    """A copy of the graph with form values and uploaded media names written into their node inputs."""
    g = copy.deepcopy(graph)
    for f in meta.get("fields", []):
        key = f["key"]
        value = media.get(key) if f["type"] in ("image", "video") else values.get(key)
        if value is None or value == "":
            continue
        for t in f["targets"]:
            node, inp = str(t[0]), t[1]
            if node not in g:
                raise BridgeError(f"Field {f['label']!r} points at node {node}, which isn't in the workflow.")
            g[node]["inputs"][inp] = value // 2 if len(t) > 2 and t[2] == "half" else value
    return g


def sampler_steps(graph: dict) -> dict[str, int]:
    """Steps each sampler node will run, as far as it can be read from the graph (0 when unknown)."""
    def num(v):
        if _is_link(v) and v[0] in graph:
            src = graph[v[0]]["inputs"]
            v = src.get("value", src.get("steps"))
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    out = {}
    for nid, n in graph.items():
        ct, i = n["class_type"], n["inputs"]
        if ct in ("KSampler", "WanVideoSampler"):
            out[nid] = int(num(i.get("steps")) or 0)
        elif ct == "KSamplerAdvanced":
            steps, start, end = num(i.get("steps")), num(i.get("start_at_step")) or 0, num(i.get("end_at_step"))
            out[nid] = int(max(0, min(end if end is not None else 10**9, steps or 0) - start)) if steps else 0
        elif ct in ("SamplerCustom", "SamplerCustomAdvanced"):
            sig = i.get("sigmas")
            out[nid] = int(num(graph[sig[0]]["inputs"].get("steps")) or 0) if _is_link(sig) and sig[0] in graph else 0
    return out


def upload(data: bytes, filename: str) -> str:
    """Put a file in ComfyUI's input folder under a content-addressed name; returns the name to use."""
    ext = Path(filename).suffix.lower() or ".png"
    name = f"obscura_{hashlib.sha256(data).hexdigest()[:20]}{ext}"
    r = _post("/upload/image", timeout=300, files={"image": (name, data)}, data={"overwrite": "true", "type": "input"})
    if r.status_code != 200:
        raise BridgeError(f"ComfyUI refused the upload of {filename} ({r.status_code}).")
    body = r.json()
    return f"{body['subfolder']}/{body['name']}" if body.get("subfolder") else body["name"]


def _error_text(body: dict) -> str:
    err = body.get("error") or {}
    msg = err.get("message") if isinstance(err, dict) else str(err)
    parts = []
    for nid, ne in (body.get("node_errors") or {}).items():
        for e in ne.get("errors", [])[:2]:
            parts.append(f"node {nid} ({ne.get('class_type')}): {e.get('message')} {e.get('details') or ''}".strip())
    return (msg or "ComfyUI rejected the workflow") + (": " + "; ".join(parts[:3]) if parts else "")


PHASES = [
    (("Loader", "LoraLoader"), "Loading models"),
    (("TextEncode",), "Reading prompt"),
    (("VisionEncode",), "Reading image"),
    (("LoadVideo", "GetVideoComponents"), "Reading video"),
    (("PoseAndFaceDetection", "PoseData"), "Detecting pose"),
    (("VAEDecode", "VideoDecode"), "Decoding frames"),
    (("CreateVideo", "SaveVideo", "VideoCombine"), "Saving video"),
    (("SaveImage",), "Saving"),
]


def _phase_for(class_type: str) -> str | None:
    if class_type in SAMPLERS:
        return "Rendering"
    for keys, label in PHASES:
        if any(k in class_type for k in keys):
            return label
    return None


def execute(graph: dict, job: dict, cancelled) -> dict:
    """Queue the graph, mirror its progress into job["step"/"phase"], and return ComfyUI's history entry.

    job["request"]["steps"] is the total sampler steps when the graph says it up front; otherwise it grows as
    samplers report. Cancelling interrupts the run (or removes it from ComfyUI's queue if it hasn't started)."""
    client_id = uuid.uuid4().hex
    steps = sampler_steps(graph)
    known_total = sum(steps.values())
    job["request"]["steps"] = known_total or 1
    sampler_ids = [n for n in _node_order(graph) if n in steps]
    done_steps = {}

    ws = None
    try:
        ws = websocket.create_connection(f"{COMFY_URL.replace('http', 'ws', 1)}/ws?clientId={client_id}", timeout=5)
        ws.settimeout(1.0)
    except (OSError, websocket.WebSocketException):
        ws = None  # progress falls back to polling /history

    r = _post("/prompt", json={"prompt": graph, "client_id": client_id}, timeout=60)
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        if ws:
            ws.close()
        raise BridgeError(_error_text(body))
    prompt_id = body["prompt_id"]
    job["comfy_prompt_id"] = prompt_id
    job["phase"] = "Waiting in ComfyUI queue"
    last_poll = 0.0
    try:
        while True:
            if cancelled():
                _post("/interrupt", json={"prompt_id": prompt_id}, timeout=5)
                _post("/queue", json={"delete": [prompt_id]}, timeout=5)
                raise Interrupted()
            msg = None
            if ws:
                try:
                    raw = ws.recv()
                    msg = json.loads(raw) if isinstance(raw, str) else None
                except websocket.WebSocketTimeoutException:
                    msg = None
                except (OSError, websocket.WebSocketException):
                    ws = None
            else:
                time.sleep(1.0)
            if msg:
                data = msg.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                kind = msg.get("type")
                if kind == "execution_start":
                    job["phase"] = "Starting"
                elif kind == "executing":
                    node = data.get("node")
                    if node is None and data.get("prompt_id") == prompt_id:
                        break
                    if node in graph:
                        phase = _phase_for(graph[node]["class_type"])
                        if phase == "Rendering" and len(sampler_ids) > 1:
                            phase = f"Rendering, pass {sampler_ids.index(node) + 1} of {len(sampler_ids)}"
                        if phase:
                            job["phase"] = phase
                elif kind == "progress" and data.get("node") in steps:
                    node, value, mx = data["node"], int(data.get("value", 0)), int(data.get("max", 0) or 1)
                    if job["step"] == 0:
                        job["first_step_at"] = time.time()
                    done_steps[node] = (value, mx)
                    total = sum(max(steps.get(n) or 0, done_steps.get(n, (0, 0))[1]) for n in sampler_ids)
                    job["request"]["steps"] = max(total, 1)
                    job["step"] = sum(v for v, _ in done_steps.values())
                elif kind == "execution_success":
                    job["phase"] = "Collecting results"  # history is written just after this message
                elif kind == "execution_error":
                    raise BridgeError(f"ComfyUI failed in {data.get('node_type', 'a node')}: {data.get('exception_message', '').strip()[:400]}")
                elif kind == "execution_interrupted":
                    raise Interrupted()
            # Safety net for missed websocket messages
            if time.time() - last_poll > (5 if ws else 1):
                last_poll = time.time()
                hist = _get(f"/history/{prompt_id}", timeout=10).json().get(prompt_id)
                if hist and (hist.get("status") or {}).get("completed"):
                    break
                if hist and (hist.get("status") or {}).get("status_str") == "error":
                    msgs = [m for m in hist["status"].get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[0][1].get("exception_message", "") if msgs else ""
                    raise BridgeError(f"ComfyUI failed: {detail.strip()[:400]}")
    finally:
        if ws:
            ws.close()
    job["step"] = job["request"]["steps"]
    job["phase"] = "Collecting results"
    # ComfyUI stores the history entry right after it reports success; wait for it to appear
    deadline = time.time() + 60
    while True:
        hist = _get(f"/history/{prompt_id}", timeout=30).json().get(prompt_id) or {}
        if (hist.get("status") or {}).get("completed") or (hist.get("status") or {}).get("status_str") == "error" or time.time() > deadline:
            break
        time.sleep(0.5)
    status_str = (hist.get("status") or {}).get("status_str")
    if status_str == "error":
        msgs = [m for m in hist["status"].get("messages", []) if m[0] == "execution_error"]
        raise BridgeError(f"ComfyUI failed: {(msgs[0][1].get('exception_message', '') if msgs else '').strip()[:400]}")
    return hist


def output_files(hist: dict) -> list[dict]:
    """Files a finished prompt saved, in node order; previews only when nothing was saved."""
    found = []
    for nid, out in sorted((hist.get("outputs") or {}).items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else 10**9)):
        for value in out.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("filename") and item.get("type") in ("output", "temp"):
                    found.append({**item, "node": nid})
    saved = [f for f in found if f["type"] == "output"]
    return saved or found


def download(item: dict, dest: Path):
    params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
    with requests.get(COMFY_URL + "/view", params=params, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# ============================================================
# Video helpers (ffmpeg / ffprobe)
# ============================================================
def probe(path: Path) -> dict:
    """Width, height, fps, frame count, duration and codec of a video's first stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,codec_name:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, ValueError, OSError):
        return {}
    s = (data.get("streams") or [{}])[0]
    num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0
    return {
        "width": s.get("width"), "height": s.get("height"), "codec": s.get("codec_name"),
        "fps": round(fps, 3) if fps else None,
        "frames": int(s["nb_read_packets"]) if str(s.get("nb_read_packets", "")).isdigit() else None,
        "duration": round(float((data.get("format") or {}).get("duration") or 0), 3) or None,
    }


def to_browser_mp4(src: Path, dest: Path):
    """Re-encode a video browsers can't play (gif, ProRes, H.265, ...) as H.264 MP4."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "17", "-preset", "medium", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags", "+faststart", "-an", str(dest)], check=True, timeout=1800)


def first_frame_png(path: Path) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "image2pipe",
                           "-vcodec", "png", "-"], capture_output=True, check=True, timeout=60).stdout
