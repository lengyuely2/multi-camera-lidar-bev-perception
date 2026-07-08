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


def _scene_indices(value: str, scene_count: int) -> list[int]:
    if value == "all":
        return list(range(scene_count))
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    unique = sorted(set(indices))
    invalid = [index for index in unique if index < 0 or index >= scene_count]
    if invalid:
        raise ValueError(f"Invalid scene indices {invalid}; dataset has {scene_count} scenes")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BEVFusion over multiple nuScenes mini scenes with one loaded model")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/bevfusion_mini/scenes"))
    parser.add_argument("--summary", type=Path, default=Path("output/bevfusion_mini/batch_summary.json"))
    parser.add_argument("--scene-indices", default="all",
                        help="all, comma list like 0,2,5, or range like 0-3")
    parser.add_argument("--max-frames-per-scene", type=int, default=0,
                        help="0 means use every frame in each selected scene")
    parser.add_argument("--score-threshold", type=float, default=0.2)
    args = parser.parse_args()

    root = args.dataroot.resolve()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    scene_indices = _scene_indices(args.scene_indices, len(nusc.scene))
    infos = mmengine.load(args.infos)
    info_by_token = {item["token"]: item for item in infos["data_list"]}

    model = init_model(str(args.config), str(args.checkpoint), device="cuda:0")
    test_pipeline = Compose(deepcopy(model.cfg.test_dataloader.dataset.pipeline))
    box_type_3d, box_mode_3d = get_box_type(model.cfg.test_dataloader.dataset.box_type_3d)
    classes = model.dataset_meta["classes"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    batch = {
        "model": "BEVFusion camera+LiDAR",
        "dataset": "nuScenes v1.0-mini",
        "scene_indices": scene_indices,
        "score_threshold": args.score_threshold,
        "max_frames_per_scene": args.max_frames_per_scene,
        "scenes": [],
    }

    for scene_position, scene_index in enumerate(scene_indices):
        scene_name, tokens = _scene_tokens(nusc, scene_index)
        if args.max_frames_per_scene > 0:
            tokens = tokens[:args.max_frames_per_scene]
        missing = [token for token in tokens if token not in info_by_token]
        if missing:
            raise RuntimeError(f"Scene {scene_name} is missing {len(missing)} tokens from info file")

        output_path = args.output_dir / f"{scene_index:02d}_{scene_name}.json"
        payload = {
            "model": "BEVFusion camera+LiDAR",
            "scene_name": scene_name,
            "scene_index": scene_index,
            "score_threshold": args.score_threshold,
            "frames": [],
        }
        print(f"Scene {scene_position + 1}/{len(scene_indices)}: {scene_name} "
              f"({len(tokens)} frames)", flush=True)

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
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  [{frame_index + 1}/{len(tokens)}] {token[:8]}: {elapsed:.3f}s, "
                  f"{payload['frames'][-1]['prediction_count']} predictions", flush=True)

        timings = np.asarray([frame["inference_seconds"] for frame in payload["frames"]])
        payload["summary"] = {
            "frames": len(payload["frames"]),
            "mean_inference_seconds": float(timings.mean()) if len(timings) else None,
            "p95_inference_seconds": float(np.percentile(timings, 95)) if len(timings) else None,
            "max_peak_gpu_memory_gib": max(
                (frame["peak_gpu_memory_gib"] for frame in payload["frames"]),
                default=None,
            ),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        batch["scenes"].append({
            "scene_index": scene_index,
            "scene_name": scene_name,
            "frames": len(payload["frames"]),
            "prediction_file": str(output_path),
            **payload["summary"],
        })
        args.summary.write_text(json.dumps(batch, indent=2), encoding="utf-8")

    all_times = [
        scene["mean_inference_seconds"]
        for scene in batch["scenes"]
        if scene["mean_inference_seconds"] is not None
    ]
    total_frames = sum(scene["frames"] for scene in batch["scenes"])
    batch["summary"] = {
        "scenes": len(batch["scenes"]),
        "frames": total_frames,
        "mean_scene_inference_seconds": float(np.mean(all_times)) if all_times else None,
        "approx_fps": float(1.0 / np.mean(all_times)) if all_times else None,
        "max_peak_gpu_memory_gib": max(
            (scene["max_peak_gpu_memory_gib"] for scene in batch["scenes"]
             if scene["max_peak_gpu_memory_gib"] is not None),
            default=None,
        ),
    }
    args.summary.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(json.dumps(batch["summary"], indent=2))
    print(f"Summary: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
