"""
Downsample LISS-4 imagery from 5.8m GSD to 10.0m GSD to match Sentinel-2.
The scaling factor is 0.58. (e.g., 14358x16226 becomes 8328x9411).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SCALE_FACTOR = 5.8 / 10.0  # 5.8m (LISS-4 MX) to 10.0m (Sentinel-2) GSD

def process_image(input_path: Path, output_path: Path) -> str:
    """Downsamples a single image dynamically using Lanczos filtering."""
    try:
        with Image.open(input_path) as img:
            target_w = int(img.width * SCALE_FACTOR)
            target_h = int(img.height * SCALE_FACTOR)
            
            resized = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
            resized.save(output_path, quality=95)
        return f"SUCCESS: {input_path.name} ({img.width}x{img.height} -> {target_w}x{target_h})"
    except Exception as e:
        return f"ERROR: {input_path.name} - {str(e)}"

def parse_args():
    p = argparse.ArgumentParser(description="Downscale 5.0m LISS-4 to 10.0m Sentinel-2 spatial resolution.")
    p.add_argument("--src_dir", required=True, help="Input directory containing BAND subfolders.")
    p.add_argument("--out_dir", required=True, help="Output directory for resized images.")
    p.add_argument("--workers", type=int, default=8, help="Number of CPU cores to use.")
    return p.parse_args()

def main():
    args = parse_args()
    src_root = Path(args.src_dir)
    out_root = Path(args.out_dir)

    if not src_root.exists():
        sys.exit(f"Source directory {src_root} not found.")

    bands = ["BAND2", "BAND3", "BAND4"]
    tasks = []

    for band in bands:
        src_band_dir = src_root / band
        out_band_dir = out_root / band
        out_band_dir.mkdir(parents=True, exist_ok=True)

        if not src_band_dir.exists():
            print(f"Warning: {src_band_dir} not found. Skipping.")
            continue

        for img_path in src_band_dir.glob("*.jpg"):
            out_path = out_band_dir / img_path.name
            tasks.append((img_path, out_path))

    print(f"Found {len(tasks)} images to resize using a {SCALE_FACTOR}x scale factor.")

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_image, src, out): src for src, out in tasks}
        
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            print(f"[{i}/{len(tasks)}] {result}")
            if result.startswith("SUCCESS"):
                success_count += 1

    print(f"\nCompleted. Successfully resized {success_count}/{len(tasks)} images.")

if __name__ == "__main__":
    main()