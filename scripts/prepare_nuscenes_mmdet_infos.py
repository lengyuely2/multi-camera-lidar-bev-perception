from __future__ import annotations

import argparse
from pathlib import Path

import mmengine

from tools.dataset_converters.nuscenes_converter import create_nuscenes_infos
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nuScenes mini infos without a training GT database")
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--prefix", default="nuscenes_mini")
    parser.add_argument("--max-sweeps", type=int, default=10)
    args = parser.parse_args()

    root = args.dataroot.resolve()
    create_nuscenes_infos(str(root), args.prefix, version="v1.0-mini", max_sweeps=args.max_sweeps)
    converted = []
    for split in ("train", "val"):
        path = root / f"{args.prefix}_infos_{split}.pkl"
        update_pkl_infos("nuscenes", out_dir=str(root), pkl_path=str(path))
        converted.append(mmengine.load(path))

    combined = {
        "metainfo": converted[0]["metainfo"],
        "data_list": converted[0]["data_list"] + converted[1]["data_list"],
    }
    combined["data_list"].sort(key=lambda item: item["timestamp"])
    output = root / f"{args.prefix}_infos_all.pkl"
    mmengine.dump(combined, output)
    print(f"Wrote {len(combined['data_list'])} samples: {output}")


if __name__ == "__main__":
    main()
