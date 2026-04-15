"""
Pre-build SWE-bench eval Docker images so they are cached locally before
running experiments with --use-eval-image.

Usage
-----
    # Build images for specific instances:
    python scripts/prebuild_images.py --instance astropy__astropy-12907 django__django-11099

    # Build images for the first N issues in the dataset:
    python scripts/prebuild_images.py --num-issues 10

    # Force rebuild even if already cached:
    python scripts/prebuild_images.py --num-issues 5 --force-rebuild
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-build SWE-bench eval Docker images")
    p.add_argument("--instance", type=str, nargs="+", default=None,
                   help="Explicit instance ID(s) to build")
    p.add_argument("--num-issues", type=int, default=None,
                   help="Build images for the first N issues in SWE-bench Lite")
    p.add_argument("--dataset-path", type=str, default="data/swe-bench-lite",
                   help="Path to local dataset (falls back to HuggingFace)")
    p.add_argument("--max-workers", type=int, default=2,
                   help="Parallel build workers (default: 2)")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Rebuild even if image already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.instance and not args.num_issues:
        print("Specify --instance <id> [id ...] or --num-issues <N>")
        sys.exit(1)

    # Resolve instance list
    if args.instance:
        instance_ids = args.instance
    else:
        from environments.swe_bench import SWEBenchEnvironment
        instance_ids = SWEBenchEnvironment.list_issues(
            dataset_path=args.dataset_path, limit=args.num_issues
        )
        print(f"Resolved {len(instance_ids)} issues from dataset")

    # Load dataset rows for the requested instances
    try:
        from datasets import load_dataset  # type: ignore[import]
        ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
        by_id = {row["instance_id"]: dict(row) for row in ds}
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        sys.exit(1)

    missing = [i for i in instance_ids if i not in by_id]
    if missing:
        print(f"Warning: {len(missing)} instance(s) not found in dataset: {missing[:5]}")
    rows = [by_id[i] for i in instance_ids if i in by_id]

    if not rows:
        print("No valid instances to build.")
        sys.exit(1)

    # Build images via swebench harness
    import docker
    from swebench.harness.docker_build import build_instance_images

    client = docker.from_env()

    print(f"\nBuilding eval images for {len(rows)} instance(s) "
          f"(max_workers={args.max_workers}, force={args.force_rebuild}) ...\n")

    build_instance_images(
        client=client,
        dataset=rows,
        force_rebuild=args.force_rebuild,
        max_workers=args.max_workers,
        namespace="swebench",
        tag="latest",
        env_image_tag="latest",
    )

    # Report which images are now available
    print("\nDone. Checking local images:")
    ok, missing_imgs = [], []
    for row in rows:
        iid = row["instance_id"]
        # Harness with namespace="swebench" replaces __ with _1776_
        remote_id = iid.lower().replace("__", "_1776_")
        img_name = f"swebench/sweb.eval.x86_64.{remote_id}:latest"
        try:
            client.images.get(img_name)
            ok.append(iid)
        except docker.errors.ImageNotFound:
            missing_imgs.append(iid)

    for iid in ok:
        print(f"  ✓  {iid}")
    for iid in missing_imgs:
        print(f"  ✗  {iid}  (image not found after build)")

    print(f"\n{len(ok)}/{len(rows)} images ready.")


if __name__ == "__main__":
    main()
