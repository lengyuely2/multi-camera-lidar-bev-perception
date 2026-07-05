from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mmdet3d.apis import inference_multi_modality_detector, init_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one BEVFusion sample without the Open3D visualizer")
    parser.add_argument("pcd", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    args = parser.parse_args()

    model = init_model(str(args.config), str(args.checkpoint), device="cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        result, _ = inference_multi_modality_detector(
            model, str(args.pcd), str(args.images), str(args.annotation), "all"
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    instances = result.pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
    scores = instances.scores_3d.detach().cpu().numpy()
    labels = instances.labels_3d.detach().cpu().numpy()
    classes = model.dataset_meta["classes"]
    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        predictions.append({
            "class": classes[int(label)],
            "score": float(score),
            "box_lidar": [float(value) for value in box],
        })

    payload = {
        "model": "BEVFusion camera+LiDAR",
        "score_threshold": args.score_threshold,
        "inference_seconds": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "prediction_count": sum(item["score"] >= args.score_threshold for item in predictions),
        "stored_prediction_count": len(predictions),
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
