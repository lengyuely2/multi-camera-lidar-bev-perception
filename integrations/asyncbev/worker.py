"""Resident AsyncBEV inference process; JSON-lines RPC, arrays in local NPZ files.

This file runs in the isolated legacy model environment, not the ROS environment.
Only the LiDAR-delay checkpoint is supported. No annotations/flow labels are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


class AsyncBEVModel:
    def __init__(self, source: Path, checkpoint: Path, score_threshold=0.2):
        import numpy as np
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model
        sys.path.insert(0, str(source))
        importlib.import_module("projects.CMT.mmdet3d_plugin")
        from projects.CMT.mmdet3d_plugin.datasets.pipelines.transform_3d import (
            ResizeCropFlipImage, NormalizeMultiviewImage, PadMultiViewImage,
        )
        spec = json.loads((ROOT / "integrations/asyncbev/upstream.json").read_text())
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != spec["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint does not match the pinned LiDAR-delay model")
        head_source = source / "projects/CMT/mmdet3d_plugin/models/dense_heads/cmt_head.py"
        if "for img_meta in img_metas if self.training else []:" not in head_source.read_text():
            raise RuntimeError("Apply the label-free inference patch using scripts/prepare_asyncbev.py")
        self.cfg = Config.fromfile(str(source / spec["config"]))
        self.cfg.model.pretrained = None
        # No pretrained initialization downloads: all learned tensors come from the checkpoint.
        for name in ("img_backbone", "pts_backbone"):
            if name in self.cfg.model:
                self.cfg.model[name].pop("init_cfg", None)
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        loaded = load_checkpoint(self.model, str(checkpoint), map_location="cpu", strict=True)
        del loaded
        self.model.cuda().eval()
        self.threshold = score_threshold
        self.resize_class = ResizeCropFlipImage
        self.normalizer = NormalizeMultiviewImage(**self.cfg.img_norm_cfg)
        self.padder = PadMultiViewImage(size_divisor=32)
        self.torch, self.np = torch, np

    def infer(self, packet):
        from pyquaternion import Quaternion
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from parking_bev.perception.sensor_packet import validate_packet
        np, torch = self.np, self.torch
        validate_packet(packet)
        images = [packet["image_" + str(index)] for index in range(6)]
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("This CMT deployment requires equal raw camera image shapes")
        augmentation = dict(self.cfg.ida_aug_conf)
        augmentation.update(H=images[0].shape[0], W=images[0].shape[1])
        intrinsic_matrices = []
        for intrinsic in packet["intrinsics"]:
            padded = np.eye(4, dtype=np.float32)
            padded[:3, :3] = intrinsic
            intrinsic_matrices.append(padded)
        data = dict(img=images, cam_intrinsic=intrinsic_matrices,
                    lidar2cam=[value.copy() for value in packet["lidar2cam"]])
        data = self.resize_class(data_aug_conf=augmentation, training=False)(data)
        shapes = [image.shape for image in data["img"]]
        data = self.padder(self.normalizer(data))
        def pose_fields(transform, prefix):
            return {prefix + "_translation": transform[:3, 3].tolist(),
                    prefix + "_rotation": Quaternion(matrix=transform[:3, :3]).elements.tolist()}
        meta = dict(
            timestamp=float(packet["timestamp_s"]),
            async_metas=dict(timestamp=float(packet["lidar_timestamp_s"]),
                             **pose_fields(packet["source_ego_to_global"], "ego2global")),
            **pose_fields(packet["ego_to_global"], "ego2global"),
            **pose_fields(packet["lidar_to_ego"], "lidar2ego"),
            lidar2img=data["lidar2img"], img_shape=shapes,
            pad_shape=[image.shape for image in data["img"]],
            box_type_3d=LiDARInstance3DBoxes,
        )
        tensor = torch.from_numpy(np.stack([image.transpose(2, 0, 1) for image in data["img"]])).float().unsqueeze(0).cuda()
        points = torch.from_numpy(np.asarray(packet["points"], dtype=np.float32)).cuda()
        valid = ((points[:, :3] >= points.new_tensor(self.cfg.point_cloud_range[:3])) &
                 (points[:, :3] < points.new_tensor(self.cfg.point_cloud_range[3:]))).all(dim=1)
        points = points[valid]
        if not len(points):
            raise ValueError("No points inside model range")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            result = self.model.simple_test([points], [meta], img=tensor)[0]["pts_bbox"]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        boxes = result["boxes_3d"]
        centers = boxes.gravity_center.detach().cpu().numpy()
        raw = boxes.tensor.detach().cpu().numpy()
        scores = result["scores_3d"].detach().cpu().numpy()
        labels = result["labels_3d"].detach().cpu().numpy()
        detections = []
        for center, box, score, label in zip(centers, raw, scores, labels):
            if score >= self.threshold:
                detections.append(dict(center=center.tolist(), size_lwh=box[3:6].tolist(),
                                       yaw=float(box[6]), velocity_xy=box[7:9].tolist(),
                                       score=float(score), class_name=self.cfg.class_names[int(label)]))
        return dict(model="CMT+AsyncBEV (label-free deployment)",
                    timestamp_s=float(packet["timestamp_s"]), frame_id=str(packet["frame_id"]),
                    inference_seconds=elapsed, detections=detections,
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/external/AsyncBEV")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/cmt_asyncbev_lidar_epoch12.pth")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--listen", help="Optional loopback host:port; packet paths must be shared with the ROS process")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    # Keep native CUDA/library stdout chatter out of the RPC channel as well.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        if args.doctor:
            import torch, mmcv, mmdet3d, flash_attn
            from mmcv.ops import nms
            sys.path.insert(0, str(args.source))
            importlib.import_module("projects.CMT.mmdet3d_plugin")
            report = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                          mmcv=mmcv.__version__, mmdet3d=mmdet3d.__version__,
                          gpu=torch.cuda.get_device_name(0), cuda_available=torch.cuda.is_available())
            # Exercise a real compiled CUDA operator, not just imports.
            nms(torch.tensor([[0., 0., 1., 1.]], device="cuda"), torch.ones(1, device="cuda"), 0.5)
            torch.cuda.synchronize()
            protocol.write(json.dumps(report) + "\n")
            return
        model = AsyncBEVModel(args.source, args.checkpoint, args.score_threshold)
        import numpy as np
        if args.packet:
            with np.load(args.packet, allow_pickle=False) as packet:
                result = model.infer(packet)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
            protocol.write(json.dumps(result) + "\n")
            return
        def serve(reader, writer):
            writer.write(json.dumps({"status": "ready", "model": "CMT+AsyncBEV"}) + "\n")
            writer.flush()
            for line in reader:
                request = json.loads(line)
                with np.load(request["packet"], allow_pickle=False) as packet:
                    result = model.infer(packet)
                writer.write(json.dumps(result, allow_nan=False) + "\n")
                writer.flush()
        if args.listen:
            host, port = args.listen.rsplit(":", 1)
            if host != "127.0.0.1":
                raise ValueError("The model service may bind only to 127.0.0.1")
            with socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((host, int(port)))
                server.listen(1)
                print("Model ready on " + args.listen, file=sys.stderr, flush=True)
                while True:
                    connection, _ = server.accept()
                    with connection, connection.makefile("r", encoding="utf-8") as reader, connection.makefile("w", encoding="utf-8") as writer:
                        serve(reader, writer)
        else:
            serve(sys.stdin, protocol)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        protocol.write(json.dumps({"error": str(exc)}) + "\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
