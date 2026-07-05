from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import mmengine
import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type
from nuscenes.nuscenes import NuScenes


def _sensor_path(root: Path, value: str, channel: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    direct = root / path
    if direct.exists():
        return str(direct)
    return str(root / "samples" / channel / path.name)


def _scene_tokens(nusc: NuScenes, scene_index: int) -> tuple[str, list[str]]:
    scene = nusc.scene[scene_index]
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = nusc.get("sample", token)["next"]
    return scene["name"], tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEVFusion over one complete nuScenes scene")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = args.dataroot.resolve()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    scene_name, tokens = _scene_tokens(nusc, args.scene_index)
    tokens = tokens[:args.max_frames]
    infos = mmengine.load(args.infos)
    info_by_token = {item["token"]: item for item in infos["data_list"]}
    missing = [token for token in tokens if token not in info_by_token]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} scene tokens from info file")

    model = init_model(str(args.config), str(args.checkpoint), device="cuda:0")
    test_pipeline = Compose(deepcopy(model.cfg.test_dataloader.dataset.pipeline))
    box_type_3d, box_mode_3d = get_box_type(model.cfg.test_dataloader.dataset.box_type_3d)
    classes = model.dataset_meta["classes"]

    payload = {
        "model": "BEVFusion camera+LiDAR",
        "scene_name": scene_name,
        "scene_index": args.scene_index,
        "score_threshold": args.score_threshold,
        "frames": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for frame_index, token in enumerate(tokens):
        info = deepcopy(info_by_token[token])
        lidar_path = _sensor_path(root, info["lidar_points"]["lidar_path"], "LIDAR_TOP")
        for channel, image_info in info["images"].items():
            image_info["img_path"] = _sensor_path(root, image_info["img_path"], channel)
        sample_input = {
            "lidar_points": {"lidar_path": lidar_path},
            "images": info["images"],
            "box_type_3d": box_type_3d,
            "box_mode_3d": box_mode_3d,
            "timestamp": info["timestamp"],
        }
        processed = test_pipeline(sample_input)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.test_step(pseudo_collate([processed]))[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        instances = result.pred_instances_3d
        boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy()
        predictions = [{
            "class": classes[int(label)],
            "score": float(score),
            "box_lidar": [float(value) for value in box],
        } for box, score, label in zip(boxes, scores, labels)]
        payload["frames"].append({
            "frame_index": frame_index,
            "sample_token": token,
            "inference_seconds": elapsed,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "prediction_count": sum(item["score"] >= args.score_threshold for item in predictions),
            "predictions": predictions,
        })
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{frame_index + 1}/{len(tokens)}] {token[:8]}: {elapsed:.3f}s, "
              f"{payload['frames'][-1]['prediction_count']} predictions", flush=True)

    timings = np.asarray([frame["inference_seconds"] for frame in payload["frames"]])
    payload["summary"] = {
        "frames": len(payload["frames"]),
        "mean_inference_seconds": float(timings.mean()),
        "p95_inference_seconds": float(np.percentile(timings, 95)),
        "max_peak_gpu_memory_gib": max(frame["peak_gpu_memory_gib"] for frame in payload["frames"]),
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
