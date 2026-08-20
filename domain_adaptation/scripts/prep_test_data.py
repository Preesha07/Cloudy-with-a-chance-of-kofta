#!/usr/bin/env python3
"""
Prep the Final/ test set (nested <Group>/<NN City>/<scene>.zip) for inference.

Mirrors run_prep_all_scenes.py, but for the hand-curated test tree where scenes
are grouped by city rather than sitting flat in data_raw/, and where output
folders should carry the city name.

Per scene:
    1. bhoonidhi_rectangle.py  <zip>  <rect_root>/<NAME>
    2. rewrite manifest.csv's scene_id -> <NAME>      (see note below)
    3. scale_liss4_data.py     --src_dir <rect_root>/<NAME> --out_dir <scaled_root>/<NAME>

<NAME> is "<NN>_<City>__<ProductID>", e.g.
    01_Itanagar__RAF21FEB2025042597011200052SSANSTUC00GTDC
Cities hold up to 2 zips, so the product ID is required to keep names unique.

The manifest rewrite matters: run_inference.py looks up shear parameters with
manifests.get(<scaled dir name>), but bhoonidhi_rectangle.py keys the manifest
by the product ID from BAND_META. Without the rewrite the lookup misses and
final_para.jpg is silently skipped.

Slicing (1_slice_liss4.py) is deliberately NOT run — run_inference.py tiles the
scaled scene itself via LazyPatchDataset. data_sliced/ is only needed by
setup_data_images.py.

Usage:
    python domain_adaptation/scripts/prep_test_data.py --group Cloud
    python domain_adaptation/scripts/prep_test_data.py --group Cloud --limit 1   # smoke test
    python domain_adaptation/scripts/prep_test_data.py --group "Cloud Free" --skip_done
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STARTERS = REPO / "domain_adaptation" / "pipeline" / "starters_for_data_prep"
TEST = REPO / "Bhoonidhi-Data" / "test_data"


def scene_name(zip_path: Path, group_root: Path) -> str:
    """<NN>_<City>__<ProductID> from <group_root>/<NN City>/<ProductID>.zip"""
    city = zip_path.parent.relative_to(group_root).as_posix()
    city = re.sub(r"[^A-Za-z0-9]+", "_", city).strip("_")
    return f"{city}__{zip_path.stem}"


def run(cmd: list[str], label: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )


def retag_manifest(manifest: Path, name: str) -> None:
    """Point the manifest's scene_id at the folder name run_inference will look up."""
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise RuntimeError(f"{manifest} has no rows — rectification produced nothing")
    for row in rows:
        row["scene_id"] = name
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def process_scene(job: tuple) -> str:
    zip_path, name, scratch, scale_workers, skip_done = job
    zip_path = Path(zip_path)
    rect_dir = TEST / "data_rect" / name
    scaled_dir = TEST / "data_scaled" / name

    bands = ("BAND2", "BAND3", "BAND4")

    # --- Step 1: rectify -----------------------------------------------------
    rect_done = (rect_dir / "manifest.csv").exists() and all(
        (rect_dir / b).exists() and any((rect_dir / b).glob("*.jpg")) for b in bands
    )
    if skip_done and rect_done:
        rect_msg = "rect SKIP"
    else:
        t = time.time()
        run(
            [sys.executable, str(STARTERS / "bhoonidhi_rectangle.py"),
             str(zip_path), str(rect_dir),
             "--scratch", str(scratch),
             "--resume", "--skip-disk-check"],
            f"rect/{name}",
        )
        retag_manifest(rect_dir / "manifest.csv", name)
        rect_msg = f"rect {time.time() - t:.0f}s"

    missing = [b for b in bands
               if not (rect_dir / b).exists() or not any((rect_dir / b).glob("*.jpg"))]
    if missing:
        raise RuntimeError(f"{name}: rectify produced no jpg for {missing}")

    # --- Step 2: scale to Sentinel-2 GSD -------------------------------------
    scaled_done = all(
        (scaled_dir / b).exists() and any((scaled_dir / b).glob("*.jpg")) for b in bands
    )
    if skip_done and scaled_done:
        scale_msg = "scale SKIP"
    else:
        t = time.time()
        run(
            [sys.executable, str(STARTERS / "scale_liss4_data.py"),
             "--src_dir", str(rect_dir),
             "--out_dir", str(scaled_dir),
             "--workers", str(scale_workers)],
            f"scale/{name}",
        )
        scale_msg = f"scale {time.time() - t:.0f}s"

    return f"OK  {name}  [{rect_msg} | {scale_msg}]"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", default="Cloud",
                   help="subfolder of Final/ to process (default: Cloud)")
    p.add_argument("--final_root", default=str(TEST / "Final"))
    p.add_argument("--scratch", default=str(TEST / "_scratch"),
                   help="unzip scratch. Defaults inside the workspace volume — do "
                        "NOT leave this on /tmp, which is a 20G overlay here.")
    p.add_argument("--scene_workers", type=int, default=4)
    p.add_argument("--scale_workers", type=int, default=4)
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N scenes (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    group_root = Path(args.final_root) / args.group
    if not group_root.is_dir():
        sys.exit(f"No such group dir: {group_root}")

    zips = sorted(group_root.rglob("*.zip"))
    if not zips:
        sys.exit(f"No zips under {group_root}")
    if args.limit:
        zips = zips[:args.limit]

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    jobs = [(str(z), scene_name(z, group_root), str(scratch),
             args.scale_workers, args.skip_done) for z in zips]

    print(f"Group: {args.group}  |  {len(jobs)} scene(s)  |  "
          f"scene_workers={args.scene_workers} scale_workers={args.scale_workers}")
    for _, name, *_ in jobs:
        print(f"  - {name}")
    print()

    t0 = time.time()
    n_ok = n_fail = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.scene_workers) as pool:
        futs = {pool.submit(process_scene, j): j[1] for j in jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                print(f"  [{n_ok + n_fail + 1}/{len(jobs)}] {fut.result()}", flush=True)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                failures.append(name)
                print(f"  [{n_ok + n_fail}/{len(jobs)}] FAIL {name}: {e}", flush=True)

    print(f"\n{n_ok}/{len(jobs)} ok in {(time.time() - t0) / 60:.1f} min")
    if failures:
        print("failed: " + ", ".join(failures))
        print("re-run with --skip_done to retry only the failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
