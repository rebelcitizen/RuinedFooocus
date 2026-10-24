# Vendored from the JobHub repository: sdk/python/jobworker/jobworker.py (keep in sync, do not edit here).
"""Generic JobHub worker: connection, registration, state flow and uploads.

Dependencies: ``websockets`` (>= 13, sync client) and ``requests``.
This file is self-contained on purpose so it can be vendored into a project
as a single module.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import queue
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

PROTOCOL_VERSION = 1

log = logging.getLogger("jobworker")


# ---------------------------------------------------------------------------
# data


@dataclass
class JobInput:
    index: int
    kind: str
    filename: str
    mime: Optional[str]
    url: str


@dataclass
class Job:
    id: str
    job_type: str
    spec: Dict[str, Any]
    inputs: List[JobInput]
    created_at: str
    attempt: int = 1

    @classmethod
    def from_message(cls, data: Dict[str, Any]) -> "Job":
        return cls(
            id=str(data["id"]),
            job_type=str(data["job_type"]),
            spec=dict(data.get("spec") or {}),
            inputs=[
                JobInput(
                    index=int(i["index"]),
                    kind=str(i.get("kind", "file")),
                    filename=str(i.get("filename", "")),
                    mime=i.get("mime"),
                    url=str(i["url"]),
                )
                for i in data.get("inputs", [])
            ],
            created_at=str(data.get("created_at", "")),
            attempt=int(data.get("attempt", 1)),
        )


class JobCancelled(Exception):
    """Raised by :meth:`JobContext.check_cancelled` when the hub cancelled the job."""


class JobContext:
    """What a running job may do: report progress, fetch inputs, add outputs."""

    def __init__(self, worker: "JobWorker", job: Job, work_dir: Path):
        self._worker = worker
        self.job = job
        self.work_dir = work_dir
        self._inputs: Dict[int, Path] = {}
        self._next_output = 0
        self._last_progress = 0.0

    # -- cancellation
    @property
    def cancelled(self) -> bool:
        return self.job.id in self._worker._cancelled

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled(self.job.id)

    # -- progress
    def progress(self, percent: float, status: Optional[str] = None, preview_path: Optional[os.PathLike] = None) -> None:
        """Report progress 0-100. Calls are throttled to about four per second."""
        now = time.monotonic()
        if percent < 100 and now - self._last_progress < 0.25:
            return
        self._last_progress = now
        self._worker._send({"type": "job.progress", "job_id": self.job.id, "percent": float(max(0, min(100, percent))), "status": status})
        if preview_path is not None:
            try:
                self._worker._upload_preview(self.job.id, Path(preview_path))
            except Exception as exc:  # previews are best effort
                log.debug("preview upload failed: %s", exc)

    # -- inputs
    def download_input(self, index: int) -> Path:
        """Download one input (cached) into the job's work directory."""
        if index in self._inputs:
            return self._inputs[index]
        inp = next((i for i in self.job.inputs if i.index == index), None)
        if inp is None:
            raise KeyError(f"job has no input {index}")
        suffix = Path(inp.filename).suffix or mimetypes.guess_extension(inp.mime or "") or ""
        target = self.work_dir / f"input-{index}{suffix}"
        self._worker._download(inp.url, target)
        self._inputs[index] = target
        return target

    def download_inputs(self) -> Dict[int, Path]:
        return {i.index: self.download_input(i.index) for i in self.job.inputs}

    # -- outputs
    def add_output(
        self,
        path: os.PathLike,
        kind: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
        index: Optional[int] = None,
        mime: Optional[str] = None,
    ) -> int:
        """Upload a result file. Returns the output index used."""
        if index is None:
            index = self._next_output
        self._next_output = max(self._next_output, index + 1)
        self._worker._upload_output(self.job.id, index, Path(path), kind, metadata or {}, mime)
        return index


# ---------------------------------------------------------------------------
# worker


class JobWorker:
    """Base class for a worker process. Override ``job_type``, ``capabilities`` and ``run``."""

    job_type: str = ""

    def __init__(
        self,
        server_url: str,
        token: str,
        display_name: str,
        version: str = "",
        work_dir: Optional[os.PathLike] = None,
        max_concurrency: int = 1,
        reconnect_min: float = 2.0,
        reconnect_max: float = 60.0,
        http_timeout: float = 120.0,
    ):
        if not self.job_type:
            raise ValueError("job_type must be set on the worker class")
        if not server_url or not token:
            raise ValueError("server_url and token are required")
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.display_name = display_name
        self.version = version
        self.max_concurrency = max_concurrency
        self.reconnect_min = reconnect_min
        self.reconnect_max = reconnect_max
        self.http_timeout = http_timeout
        self.work_root = Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "jobworker"
        self.work_root.mkdir(parents=True, exist_ok=True)

        self._ws = None
        self._ws_lock = threading.Lock()
        self._pending: List[Dict[str, Any]] = []
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._cancelled: set = set()
        self._current_job_id: Optional[str] = None
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.worker_id: Optional[str] = None
        self.connected = threading.Event()
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self.token}"

    # -- to override ---------------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        return {}

    def run(self, job: Job, ctx: JobContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def on_cancel(self, job_id: str) -> None:
        """Hook called when the hub cancels a job (also sets ctx.cancelled)."""

    # -- lifecycle -------------------------------------------------------------------
    def start(self) -> List[threading.Thread]:
        """Start the connection and runner threads (daemon) and return them."""
        if self._threads:
            return self._threads
        self._stop.clear()
        conn = threading.Thread(target=self._connection_loop, name="jobworker-conn", daemon=True)
        runner = threading.Thread(target=self._runner_loop, name="jobworker-run", daemon=True)
        self._threads = [conn, runner]
        conn.start()
        runner.start()
        return self._threads

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    # -- connection ------------------------------------------------------------------
    @property
    def ws_url(self) -> str:
        base = self.server_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return base + "/ws/worker"

    def _connection_loop(self) -> None:
        delay = self.reconnect_min
        while not self._stop.is_set():
            try:
                log.info("connecting to %s", self.ws_url)
                with connect(
                    self.ws_url,
                    additional_headers={"Authorization": f"Bearer {self.token}"},
                    open_timeout=15,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=4 * 1024 * 1024,
                ) as ws:
                    with self._ws_lock:
                        self._ws = ws
                    hello = {
                        "type": "hello",
                        "protocol": PROTOCOL_VERSION,
                        "display_name": self.display_name,
                        "job_type": self.job_type,
                        "version": self.version,
                        "capabilities": self.capabilities(),
                        "max_concurrency": self.max_concurrency,
                    }
                    if self._current_job_id:
                        hello["current_job_id"] = self._current_job_id
                    ws.send(json.dumps(hello))
                    first = json.loads(ws.recv(timeout=15))
                    if first.get("type") != "hello_ok":
                        raise RuntimeError(f"registration rejected: {first.get('message') or first}")
                    self.worker_id = first.get("worker_id")
                    log.info("registered as worker %s (%s)", self.worker_id, self.display_name)
                    delay = self.reconnect_min
                    self.connected.set()
                    self._flush_pending()
                    for raw in ws:
                        self._handle_message(raw)
            except ConnectionClosed as exc:
                log.warning("connection closed: %s", exc)
            except Exception as exc:
                log.warning("connection error: %s", exc)
            finally:
                self.connected.clear()
                with self._ws_lock:
                    self._ws = None
            if self._stop.is_set():
                break
            log.info("reconnecting in %.0fs", delay)
            self._stop.wait(delay)
            delay = min(delay * 2, self.reconnect_max)

    def _handle_message(self, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            log.warning("ignoring non-JSON message")
            return
        kind = msg.get("type")
        if kind == "job.assign":
            job = Job.from_message(msg["job"])
            log.info("job %s assigned (attempt %d)", job.id, job.attempt)
            self._queue.put(job)
        elif kind == "job.cancel":
            job_id = str(msg.get("job_id"))
            log.info("job %s cancel requested", job_id)
            self._cancelled.add(job_id)
            try:
                self.on_cancel(job_id)
            except Exception as exc:
                log.warning("on_cancel raised: %s", exc)
        elif kind == "error":
            log.warning("hub error: %s", msg.get("message"))
        elif kind == "hello_ok":
            pass
        else:
            log.debug("unknown message type %s", kind)

    def _send(self, msg: Dict[str, Any]) -> None:
        data = json.dumps(msg)
        with self._ws_lock:
            ws = self._ws
            if ws is None:
                self._pending.append(msg)
                return
            try:
                ws.send(data)
            except Exception as exc:
                log.warning("send failed (%s); queued for retry", exc)
                self._pending.append(msg)

    def _flush_pending(self) -> None:
        with self._ws_lock:
            pending, self._pending = self._pending, []
            ws = self._ws
        for msg in pending:
            if msg.get("type") == "job.progress":
                continue  # stale
            try:
                ws.send(json.dumps(msg))
            except Exception:
                with self._ws_lock:
                    self._pending.append(msg)
                return

    # -- running jobs -------------------------------------------------------------------
    def _runner_loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        if job.id in self._cancelled:
            self._send({"type": "job.failed", "job_id": job.id, "error": "cancelled before start"})
            return
        self._current_job_id = job.id
        work_dir = self.work_root / job.id
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = JobContext(self, job, work_dir)
        try:
            self._send({"type": "job.progress", "job_id": job.id, "percent": 0.0, "status": "starting"})
            self.run(job, ctx)
            if ctx.cancelled:
                self._send({"type": "job.failed", "job_id": job.id, "error": "cancelled"})
            else:
                self._send({"type": "job.done", "job_id": job.id, "outputs": ctx._next_output})
                log.info("job %s done (%d outputs)", job.id, ctx._next_output)
        except JobCancelled:
            self._send({"type": "job.failed", "job_id": job.id, "error": "cancelled"})
        except Exception as exc:
            log.exception("job %s failed", job.id)
            self._send({"type": "job.failed", "job_id": job.id, "error": f"{type(exc).__name__}: {exc}"[:4000]})
        finally:
            self._current_job_id = None
            self._cancelled.discard(job.id)
            shutil.rmtree(work_dir, ignore_errors=True)

    # -- http ---------------------------------------------------------------------------
    def _download(self, url: str, target: Path) -> None:
        with self._session.get(url, stream=True, timeout=self.http_timeout) as resp:
            resp.raise_for_status()
            with open(target, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)

    def _upload_output(self, job_id: str, index: int, path: Path, kind: str, metadata: Dict[str, Any], mime: Optional[str]) -> None:
        mime = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        url = f"{self.server_url}/api/worker/jobs/{job_id}/outputs/{index}"
        with open(path, "rb") as fh:
            resp = self._session.put(
                url,
                files={"file": (path.name, fh, mime)},
                data={"kind": kind, "metadata": json.dumps(metadata)},
                timeout=self.http_timeout,
            )
        if resp.status_code >= 300:
            raise RuntimeError(f"output upload failed ({resp.status_code}): {resp.text[:300]}")

    def _upload_preview(self, job_id: str, path: Path) -> None:
        url = f"{self.server_url}/api/worker/jobs/{job_id}/preview"
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        with open(path, "rb") as fh:
            resp = self._session.put(url, files={"file": (path.name, fh, mime)}, timeout=30)
        if resp.status_code >= 300:
            raise RuntimeError(f"preview upload failed ({resp.status_code})")


# ---------------------------------------------------------------------------
# configuration helper


def load_env_file(path: os.PathLike, override: bool = False) -> Dict[str, str]:
    """Read KEY=VALUE lines from a .env file into os.environ (no dependency on python-dotenv).

    Lines starting with # are ignored; surrounding quotes are stripped. Existing
    environment variables win unless ``override`` is true. Returns the values read.
    """
    values: Dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values
