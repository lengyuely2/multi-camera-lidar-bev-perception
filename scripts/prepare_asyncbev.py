"""Fetch the pinned upstream source and checksum-verified LiDAR-delay checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / "integrations/asyncbev/upstream.json").read_text())
    checkout = ROOT / "data/external/AsyncBEV"
    if not checkout.exists():
        subprocess.run(["git", "clone", "--no-checkout", spec["repository"], str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", spec["revision"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["revision"]:
        raise RuntimeError("Upstream revision differs from integrations/asyncbev/upstream.json; preserve and inspect the checkout first")
    patch = ROOT / "integrations/asyncbev/label_free_inference.patch"
    check = subprocess.run(["git", "-C", str(checkout), "apply", "--check", str(patch)], capture_output=True)
    if check.returncode == 0:
        subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
    else:
        subprocess.run(["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)], check=True)
    if args.source_only:
        return
    target = ROOT / "data/checkpoints" / spec["checkpoint"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != spec["checkpoint_sha256"]:
            raise RuntimeError("Existing checkpoint hash mismatch; file left untouched")
        print("Checkpoint already verified:", target)
        return
    partial = target.with_suffix(".pth.partial")
    request = urllib.request.Request(spec["checkpoint_url"], headers={"User-Agent": "parking-bev/asyncbev-setup"})
    print("Downloading checkpoint (440 MB)...", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as destination:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            destination.write(chunk)
    if partial.stat().st_size != spec["checkpoint_size_bytes"] or sha256(partial) != spec["checkpoint_sha256"]:
        raise RuntimeError("Downloaded checkpoint failed size/SHA-256 verification; partial retained for diagnosis")
    partial.replace(target)
    print("Checkpoint SHA-256 verified:", target)


if __name__ == "__main__":
    main()
