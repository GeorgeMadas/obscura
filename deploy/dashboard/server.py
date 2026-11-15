"""Obscura setup dashboard: live view of provisioning, downloads, network, disk, GPU and services.

Stdlib only (runs on the system python3), so it works before any package or model is installed.
Started at the top of deploy/provision.sh as the supervisor service "obscura-dashboard", behind Caddy.

Nothing about the models is hardcoded. Downloads are found by watching:
  - the Hugging Face cache ($HF_HOME/hub/<repo>/blobs/*.incomplete, snapshots/ for finished files)
  - the watched folders (WATCH_DIRS): *.incomplete files from hf_hub_download(local_dir=...) staging
    and any file that is still growing (plain wget/curl downloads)
  - "Fetching <org>/<repo>/..." lines in the setup log, which name repos that have no cache folder yet
Each file's total size comes from the Hub API (matched by its etag, which hf_hub_download puts in the
in-progress file name), so there is a real progress bar and ETA wherever the Hub knows the file.

Never serves files from disk and never returns environment values: secret-looking env values, tokens
and credentials in URLs are masked in everything that leaves this process.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", int(os.environ.get("DASHBOARD_PORT", "16060"))
HERE = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE") or "/workspace")
HF_HUB = Path(os.environ.get("HF_HOME") or WORKSPACE / ".hf_home") / "hub"
WATCH_DIRS = [WORKSPACE / "models", WORKSPACE / "qwen-image" / "upscalers"]
LOG = Path("/var/log/obscura-provision.log")
DONE_MARKER = WORKSPACE / ".obscura-provisioned"
RUN_START = "Vast entrypoint is up, provisioning"
RUN_DONE = "Provisioning complete"
SERVICES = {"qwen-image": 17860, "comfyui": 18188}  # supervisor program -> internal port
SMALL = 1 << 20  # files under 1 MB (configs, tokenizers) are counted but not listed one by one

# ---------------------------------------------------------------- secret masking
_SECRET_KEYS = re.compile(r"TOKEN|KEY|SECRET|PASS|AUTH|CREDENTIAL", re.I)
_SECRETS = sorted({v for k, v in os.environ.items() if _SECRET_KEYS.search(k) and len(v) >= 6},
                  key=len, reverse=True)
_PATTERNS = [
    (re.compile(r"(https?://)[^/\s:@]+:[^@\s]+@"), r"\1•••@"),
    (re.compile(r"\b(ghp|gho|ghu|ghs|ghr|github_pat|hf|sk-ant|sk)_[A-Za-z0-9_\-]{8,}"), "•••"),
    (re.compile(r"(?i)(authorization:\s*(bearer|token)\s+)\S+"), r"\1•••"),
    (re.compile(r"(?i)\b([A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API_KEY)[A-Z0-9_]*=)\S+"), r"\1•••"),
]


def mask(s):
    for v in _SECRETS:
        if v in s:
            s = s.replace(v, "•••")
    for rx, rep in _PATTERNS:
        s = rx.sub(rep, s)
    return s


# ---------------------------------------------------------------- Hub metadata (sizes by etag)
class HubIndex:
    """etag -> (repo, path, size) and (repo, path) -> size, filled in the background from the Hub API."""

    def __init__(self):
        self.by_etag, self.by_path = {}, {}
        self.done, self.retry_at, self.queue = set(), {}, []
        self.lock, self.wake = threading.Lock(), threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def want(self, repo, rev="main"):
        key = (repo, rev)
        with self.lock:
            if key in self.done or key in self.queue or time.time() < self.retry_at.get(key, 0):
                return
            self.queue.append(key)
        self.wake.set()

    def _worker(self):
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        while True:
            self.wake.wait(5)
            self.wake.clear()
            while True:
                with self.lock:
                    if not self.queue:
                        break
                    repo, rev = self.queue[0]
                url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
                req = urllib.request.Request(url, headers={"User-Agent": "obscura-setup-dashboard"})
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                try:
                    with urllib.request.urlopen(req, timeout=15) as r:
                        info = json.load(r)
                    with self.lock:
                        for s in info.get("siblings", []):
                            size = (s.get("lfs") or {}).get("size") or s.get("size")
                            entry = (repo, s["rfilename"], size)
                            self.by_path[(repo, s["rfilename"])] = size
                            for etag in ((s.get("lfs") or {}).get("sha256"), s.get("blobId")):
                                if etag:
                                    self.by_etag[etag] = entry
                        self.done.add((repo, rev))
                except Exception:
                    with self.lock:
                        self.retry_at[(repo, rev)] = time.time() + 60
                with self.lock:
                    self.queue.pop(0)

    def etag(self, etag):
        with self.lock:
            return self.by_etag.get(etag)

    def match_name(self, name):
        """Look a plain file name up across every indexed repo (for wget-style downloads)."""
        with self.lock:
            hits = [(repo, p, size) for (repo, p), size in self.by_path.items() if p.rsplit("/", 1)[-1] == name]
        return hits[0] if len(hits) == 1 else None


HUB = HubIndex()


# ---------------------------------------------------------------- setup log
_PHASE = re.compile(r"^\[(obscura|comfyui) (\d\d):(\d\d):(\d\d)\] (.*)$")
_FETCH = re.compile(r"Fetching ([A-Za-z0-9][\w.\-]*/[\w.\-]+)")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r")


def provision_running():
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes()
        except OSError:
            continue
        if b"provision.sh" in cmd and b"bash" in cmd:
            return True
    return False


def read_log():
    """Lines of the current run only (the log is appended to on every boot)."""
    try:
        size = LOG.stat().st_size
        with LOG.open("rb") as f:
            f.seek(max(0, size - 4_000_000))
            text = f.read().decode("utf-8", "replace")
        mtime = LOG.stat().st_mtime
    except OSError:
        return [], None
    lines = [_ANSI.sub("", l) for l in text.split("\n")]
    start = 0
    for i in range(len(lines) - 1, -1, -1):
        if RUN_START in lines[i]:
            # keep the "waiting" lines just before it (nothing is logged before it today)
            start = i
            break
    return [l for l in lines[start:] if l.strip()], mtime


def parse_setup(lines, log_mtime, now):
    phases, failures, repos = [], [], []
    day_offset, last_secs = 0, None
    for line in lines:
        m = _PHASE.match(line)
        if m:
            src, hh, mm, ss, msg = m.groups()
            secs = int(hh) * 3600 + int(mm) * 60 + int(ss)
            if last_secs is not None and secs + 3600 < last_secs:
                day_offset += 86400  # crossed midnight
            last_secs = secs
            phases.append({"source": src, "t": secs + day_offset, "clock": f"{hh}:{mm}:{ss}", "text": mask(msg)})
        if "FAILED" in line:
            failures.append(mask(line.strip()))
        for repo in _FETCH.findall(line):
            parts = repo.split("@")[0].split("/")
            repos.append("/".join(parts[:2]))
    running = provision_running()
    complete = any(RUN_DONE in p["text"] for p in phases)
    if not complete and not running and DONE_MARKER.exists() and log_mtime and DONE_MARKER.stat().st_mtime >= log_mtime - 5:
        complete = True
    # The log only has clock times: anchor the last phase line to the latest matching moment no later than
    # the log's last write, then lay the others out from their offsets (handles runs that cross midnight).
    started_at = None
    if phases:
        ref = log_mtime or now
        lt = time.localtime(ref)
        midnight = ref - (lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec)
        last_epoch = midnight + phases[-1]["t"] % 86400
        if last_epoch > ref + 60:
            last_epoch -= 86400
        started_at = last_epoch - (phases[-1]["t"] - phases[0]["t"])
        for p in phases:
            p["at"] = started_at + (p["t"] - phases[0]["t"])
    if complete:
        state = "failed" if failures else "done"
    elif running:
        state = "running"
    elif not lines:
        state = "waiting"
    else:
        state = "stopped"  # script exited without the completion line (e.g. clone FAILED)
    live = state == "running"
    for i, p in enumerate(phases):
        end = phases[i + 1]["at"] if i + 1 < len(phases) else (now if live else p["at"])
        p["duration"] = max(0, end - p["at"])
        del p["t"]
    elapsed = ((now if live else phases[-1]["at"]) - started_at) if phases else 0
    return {
        "state": state, "phases": phases, "failures": failures[-20:], "elapsed": max(0, elapsed),
        "started_at": started_at,
        "current": phases[-1]["text"] if phases else None,
        "log": [mask(l)[:400] for l in lines[-80:]],
    }, set(repos)


# ---------------------------------------------------------------- downloads
class Downloads:
    """Remembers every file seen so speed, ETA and "finished during this run" survive between scans."""

    def __init__(self):
        self.files = {}  # key -> record
        self.boot = time.time()

    def _update(self, key, now, **rec):
        cur = self.files.get(key)
        have = rec["have"]
        if cur is None:
            cur = self.files[key] = {"first_seen": now, "speed": 0.0, "t": now, "have": have,
                                     "observed_active": False, "history": []}
        dt = now - cur["t"]
        if dt > 0:
            inst = max(0, have - cur["have"]) / dt
            cur["speed"] = inst if cur["speed"] == 0 else 0.6 * cur["speed"] + 0.4 * inst
        cur.update(rec, t=now)
        if rec["status"] == "downloading":
            cur["observed_active"] = True
            cur["last_active"] = now
        cur["seen"] = now
        return cur

    def scan(self, now, run_start):
        seen = set()
        # 1. Hugging Face cache
        if HF_HUB.is_dir():
            for repo_dir in HF_HUB.iterdir():
                if not repo_dir.name.startswith(("models--", "datasets--", "spaces--")):
                    continue
                repo = "/".join(repo_dir.name.split("--")[1:])
                snaps = repo_dir / "snapshots"
                revs = [p.name for p in snaps.iterdir()] if snaps.is_dir() else []
                for rev in revs or ["main"]:
                    HUB.want(repo, rev)
                names = {}
                if snaps.is_dir():
                    for root, _, fnames in os.walk(snaps):
                        for fn in fnames:
                            link = Path(root) / fn
                            try:
                                names[os.path.basename(os.readlink(link))] = str(link.relative_to(snaps)).split("/", 1)[1]
                            except (OSError, IndexError, ValueError):
                                pass
                blobs = repo_dir / "blobs"
                if not blobs.is_dir():
                    continue
                for b in blobs.iterdir():
                    try:
                        st = b.stat()
                    except OSError:
                        continue
                    incomplete = b.name.endswith(".incomplete")
                    etag = b.name[:-len(".incomplete")] if incomplete else b.name
                    meta = HUB.etag(etag)
                    name = names.get(etag) or (meta[1] if meta else etag[:12] + "…")
                    total = meta[2] if meta else (None if incomplete else st.st_size)
                    key = f"hf:{repo}:{etag}"
                    seen.add(key)
                    have = _bytes(st) if incomplete else st.st_size
                    self._update(key, now, name=name, where=repo, kind="Hugging Face cache", have=have,
                                 total=total, mtime=st.st_mtime,
                                 status="downloading" if incomplete else "done")
        # 2. Watched folders: hf local_dir staging (*.incomplete) and growing plain files
        for base in WATCH_DIRS:
            if not base.is_dir():
                continue
            for root, dirs, fnames in os.walk(base):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
                for fn in fnames:
                    p = Path(root) / fn
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    rel = p.relative_to(base)
                    staged = ".download" in rel.parts
                    if fn.endswith(".incomplete"):
                        # hf local_dir staging: "<hash of name>.<etag>[.<suffix>].incomplete"
                        m = _ETAG.search(fn)
                        meta = HUB.etag(m.group(1)) if m else None
                        name = meta[1].rsplit("/", 1)[-1] if meta else fn
                        key = f"dir:{base}:{name}"
                        seen.add(key)
                        self._update(key, now, name=name, where=(meta[0] if meta else base.name),
                                     kind=f"{base.name} (staging)", have=_bytes(st),
                                     total=meta[2] if meta else None, mtime=st.st_mtime, status="downloading")
                        continue
                    if ".cache" in rel.parts:
                        continue  # the Hub's bookkeeping (locks, metadata, CACHEDIR.TAG)
                    key = f"dir:{base}:{fn}"
                    prev = self.files.get(key)
                    growing = prev is not None and (st.st_size != prev["have"] or (
                        prev["status"] == "downloading" and now - st.st_mtime < 3))
                    recent_write = now - st.st_mtime < 3 and st.st_size > 0
                    total = prev["total"] if prev and prev.get("total") else None
                    if (growing or recent_write) and total is None and not staged:
                        hit = HUB.match_name(fn)
                        total = hit[2] if hit else None
                    status = "downloading" if (growing or recent_write) and (total is None or st.st_size < total) else "done"
                    if staged and status == "done":
                        status = "staged"  # downloaded, about to be moved into its models folder
                    where = str(rel.parent) if str(rel.parent) != "." else base.name
                    seen.add(key)
                    self._update(key, now, name=fn, where=where if not staged else "staging",
                                 kind=base.name, have=st.st_size,
                                 total=total or (st.st_size if status != "downloading" else None),
                                 mtime=st.st_mtime, status=status)
        # forget files that disappeared (moved into place / deleted); a staged file lands under a new key
        for key in [k for k in self.files if k not in seen]:
            del self.files[key]
        return self._report(now, run_start)

    def _report(self, now, run_start):
        rows, small_count, small_bytes = [], 0, 0
        for key, f in self.files.items():
            have, total = f["have"], f["total"]
            if f["status"] == "downloading":
                status = "downloading" if now - f["mtime"] < 30 else "stalled"
                speed = f["speed"] if status == "downloading" else 0
                eta = (total - have) / speed if total and speed > 1024 else None
            else:
                fresh = f["observed_active"] or (run_start and f["mtime"] >= run_start)
                status = f["status"] if f["status"] == "staged" else ("done" if fresh else "cached")
                speed, eta = 0, None
            if status != "downloading" and status != "stalled" and (total or have) < SMALL:
                small_count += 1
                small_bytes += total or have
                continue
            rows.append({
                "id": key, "name": mask(f["name"]), "where": mask(f["where"]), "kind": f["kind"],
                "have": have, "total": total, "speed": speed, "eta": eta, "status": status,
                "finished": f["mtime"] if status in ("done", "staged", "cached") else None,
            })
        order = {"downloading": 0, "stalled": 1, "staged": 2, "done": 3, "cached": 4}
        rows.sort(key=lambda r: (order[r["status"]], -(r["finished"] or 0), r["name"]))
        active = [r for r in rows if r["status"] in ("downloading", "stalled")]
        return {
            "rows": rows, "small": {"count": small_count, "bytes": small_bytes},
            "active": len(active),
            "active_bytes": sum(r["have"] for r in active),
            "active_total": sum(r["total"] or r["have"] for r in active),
            "speed": sum(r["speed"] for r in active),
        }


_ETAG = re.compile(r"(?:^|\.)([0-9a-f]{64}|[0-9a-f]{40})(?=\.)")


def _bytes(st):
    # hf_transfer writes chunks at offsets, so the apparent size can run ahead; allocated blocks don't
    return min(st.st_size, st.st_blocks * 512) if st.st_blocks else st.st_size


# ---------------------------------------------------------------- machine
class Net:
    def __init__(self):
        self.prev, self.history, self.rate = None, [], (0.0, 0.0)
        self.first = None

    def sample(self, now):
        rx = tx = 0
        try:
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
                iface, data = line.split(":", 1)
                if iface.strip() == "lo":
                    continue
                f = data.split()
                rx += int(f[0])
                tx += int(f[8])
        except OSError:
            return
        if self.first is None:
            self.first = rx
        if self.prev:
            dt = now - self.prev[0]
            if dt > 0:
                self.rate = (max(0, rx - self.prev[1]) / dt, max(0, tx - self.prev[2]) / dt)
                self.history.append([round(now), round(self.rate[0])])
                self.history = self.history[-300:]
        self.prev = (now, rx, tx)

    def report(self):
        return {"rx": self.rate[0], "tx": self.rate[1], "history": self.history,
                "rx_total": (self.prev[1] - self.first) if self.prev else 0,
                "peak": max((h[1] for h in self.history), default=0)}


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        num = lambda x: float(x) if re.match(r"^[\d.]+$", x) else None  # noqa: E731
        gpus.append({"name": f[0], "util": num(f[1]), "mem_used": num(f[2]), "mem_total": num(f[3]),
                     "temp": num(f[4]), "power": num(f[5]), "power_limit": num(f[6])})
    return gpus


def services():
    try:
        out = subprocess.run(["supervisorctl", "status", *SERVICES], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        out = ""
    res = []
    for name, port in SERVICES.items():
        m = re.search(rf"^{re.escape(name)}\s+(\S+)\s*(.*)$", out, re.M)
        state = m.group(1) if m else "UNKNOWN"
        if "no such process" in (m.group(0) if m else ""):
            state = "NOT INSTALLED"
        detail = m.group(2).strip() if m and state != "NOT INSTALLED" else ""
        uptime = re.search(r"uptime ([\d:]+(?: days?, [\d:]+)?)", detail)
        listening = False
        if state == "RUNNING":
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    listening = True
            except OSError:
                pass
        res.append({"name": name, "state": state, "listening": listening,
                    "uptime": uptime.group(1) if uptime else None})
    return res


def disk():
    try:
        u = shutil.disk_usage(WORKSPACE)
        return {"total": u.total, "used": u.used, "free": u.free}
    except OSError:
        return None


# ---------------------------------------------------------------- sampler
STATE = {"json": b"{}"}


def sampler():
    downloads, net = Downloads(), Net()
    slow = {"gpu": [], "services": [], "disk": None, "at": 0}
    while True:
        now = time.time()
        try:
            net.sample(now)
            lines, log_mtime = read_log()
            setup, repos = parse_setup(lines, log_mtime, now)
            for repo in repos:
                HUB.want(repo)
            run_start = setup["started_at"]
            dl = downloads.scan(now, run_start)
            if now - slow["at"] >= 2:
                slow.update(gpu=gpu(), services=services(), disk=disk(), at=now)
            STATE["json"] = json.dumps({
                "now": now, "setup": setup, "downloads": dl, "net": net.report(),
                "gpu": slow["gpu"], "services": slow["services"], "disk": slow["disk"],
            }, separators=(",", ":")).encode()
        except Exception as e:  # keep the dashboard alive whatever the disk looks like
            print("sampler error:", mask(repr(e)), flush=True)
        time.sleep(max(0.2, 1 - (time.time() - now)))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            self._send(200, "application/json", STATE["json"])
        elif path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (HERE / "index.html").read_bytes())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    print(f"Obscura setup dashboard on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
