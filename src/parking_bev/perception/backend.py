"""Model-independent metric detections and a bounded, isolated model worker."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

from .predictions import Prediction3D, category_for_detection_class
from ..sensors.nuscenes_source import Object3D


@dataclass(frozen=True)
class DetectionBatch:
    timestamp_s: float
    frame_id: str
    predictions: tuple[Prediction3D, ...]
    model: str
    inference_seconds: float
    peak_gpu_memory_gib: float | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> "DetectionBatch":
        """Contract: gravity centers, L/W/H in m, yaw radians, XY velocity m/s."""
        items = []
        for index, record in enumerate(payload["detections"]):
            center = np.asarray(record["center"], dtype=np.float64)
            size = np.asarray(record["size_lwh"], dtype=np.float64)
            velocity = np.asarray(record["velocity_xy"], dtype=np.float64)
            yaw, score = float(record["yaw"]), float(record["score"])
            if center.shape != (3,) or size.shape != (3,) or velocity.shape != (2,):
                raise ValueError("Invalid detection dimensions")
            if not all(np.isfinite(value).all() for value in (center, size, velocity, yaw, score)):
                raise ValueError("Non-finite detection")
            if np.any(size <= 0) or not 0 <= score <= 1:
                raise ValueError("Invalid detection size or score")
            name = str(record["class_name"])
            obj = Object3D(f"detection-{index}", category_for_detection_class(name), center,
                           size[[1, 0, 2]], yaw, velocity)
            items.append(Prediction3D(obj, name, score))
        timestamp, elapsed = float(payload["timestamp_s"]), float(payload["inference_seconds"])
        if not np.isfinite([timestamp, elapsed]).all() or elapsed < 0 or not payload["frame_id"]:
            raise ValueError("Invalid batch metadata")
        peak = payload.get("peak_gpu_memory_gib")
        if peak is not None and (not np.isfinite(float(peak)) or float(peak) < 0):
            raise ValueError("Invalid GPU memory measurement")
        return cls(timestamp, str(payload["frame_id"]), tuple(items), str(payload["model"]), elapsed,
                   float(peak) if peak is not None else None)


def transform_predictions(predictions: tuple[Prediction3D, ...], transform: np.ndarray) -> list[Prediction3D]:
    """Rigid transform from the detection frame to the consumer's frame."""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    transformed = []
    for prediction in predictions:
        obj = prediction.object
        heading = rotation @ np.array([np.cos(obj.yaw_ego), np.sin(obj.yaw_ego), 0.0])
        transformed.append(Prediction3D(Object3D(
            obj.token, obj.category, rotation @ obj.center_ego + translation, obj.size_wlh,
            float(np.arctan2(heading[1], heading[0])), (rotation @ np.r_[obj.velocity_ego, 0.0])[:2],
        ), prediction.class_name, prediction.score))
    return transformed


class ModelWorker:
    """One resident process. Timeout/protocol failure closes it; never silently substitutes detections."""

    def __init__(self, command: list[str] | None = None, startup_timeout_s: float = 180, timeout_s: float = 30,
                 packet_directory: str | None = None, endpoint: tuple[str, int] | None = None):
        if bool(command) == bool(endpoint):
            raise ValueError("Provide exactly one worker command or loopback endpoint")
        self.timeout_s = timeout_s
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        if packet_directory:
            Path(packet_directory).mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="parking-bev-", dir=packet_directory)
        self._process = None
        self._socket = None
        self._closed = False
        try:
            if endpoint:
                if endpoint[0] not in ("127.0.0.1", "localhost", "::1"):
                    raise ValueError("Only loopback model endpoints are supported")
                deadline = time.monotonic() + startup_timeout_s
                while True:
                    try:
                        self._socket = socket.create_connection(endpoint, timeout=1)
                        self._socket.settimeout(None)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Cannot connect to model worker")
                        time.sleep(0.2)
                self._input = self._socket.makefile("w", encoding="utf-8")
                self._output = self._socket.makefile("r", encoding="utf-8")
            else:
                self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 text=True, encoding="utf-8", bufsize=1)
                self._input, self._output = self._process.stdin, self._process.stdout
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            message = self._receive(startup_timeout_s)
            if message.get("status") != "ready":
                raise RuntimeError(f"Model initialization failed: {message}")
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        try:
            for line in self._output:
                self._responses.put(json.loads(line))
        except Exception as exc:
            self._responses.put(exc)
        finally:
            self._responses.put(RuntimeError("Model worker exited"))

    def _receive(self, timeout: float) -> dict:
        try:
            value = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Model worker timed out") from exc
        if isinstance(value, Exception):
            raise value
        return value

    def infer(self, packet: dict[str, np.ndarray]) -> DetectionBatch:
        with self._lock:
            if not self.alive:
                raise RuntimeError("Model worker is not running")
            packet_path = Path(self._temporary.name) / "packet.npz"
            np.savez(packet_path, **packet)
            try:
                self._input.write(json.dumps({"packet": str(packet_path)}) + "\n")
                self._input.flush()
                result = self._receive(self.timeout_s)
                if "error" in result:
                    raise RuntimeError(result["error"])
                batch = DetectionBatch.from_payload(result)
                if batch.frame_id != str(packet["frame_id"]) or batch.timestamp_s != float(packet["timestamp_s"]):
                    raise RuntimeError("Worker returned the wrong sensor frame/time")
                return batch
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        connection = getattr(self, "_socket", None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for name in ("_input", "_output"):
            stream = getattr(self, name, None)
            if stream:
                stream.close()
        self._temporary.cleanup()

    @property
    def alive(self) -> bool:
        return not self._closed and (self._process is None or self._process.poll() is None) and self._reader.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
