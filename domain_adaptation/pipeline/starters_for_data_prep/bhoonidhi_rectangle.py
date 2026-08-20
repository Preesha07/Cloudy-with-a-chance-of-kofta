#!/usr/bin/env python3
"""
bhoonidhi_rectangle_split.py  --  Bhoonidhi/NRSC zips -> per-band rectangular GeoTIFFs.

Writes ONE single-band GeoTIFF per input band routed into specific subdirectories:
    .../target_A/BAND2/<scene_id>_rect.tif
    .../target_A/BAND3/<scene_id>_rect.tif
    .../target_A/BAND4/<scene_id>_rect.tif
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from PIL import Image

GB = 1 << 30
LOG = logging.getLogger("rect")

MANIFEST_COLS = [
    "scene_id", "out_files", "jpg_files", "source_zip", "satellite", "sensor", "date",
    "path", "row", "bands", "epsg", "method",
    "in_width", "in_height", "out_width", "out_height",
    "jpg_width", "jpg_height",
    "valid_pct_of_grid", "kept_pct_of_valid",
    "shear_col_per_row", "shear_deg", "col_step_m", "row_step_m",
    "dev_max_px", "dev_mean_px", "ulx", "uly", "out_bytes", "seconds",
]


# ==========================================================================
# product discovery 
# ==========================================================================
def parse_meta(path: Path) -> dict:
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def band_number(p: Path) -> int:
    m = re.search(r"(?:BAND|B)[_-]?(\d+)", p.name, re.IGNORECASE)
    return int(m.group(1)) if m else 999


def find_bands(folder: Path) -> list[Path]:
    tifs = [f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in (".tif", ".tiff")
            and re.search(r"(?:BAND|B)[_-]?\d", f.name, re.IGNORECASE)]
    return sorted(tifs, key=band_number)


def find_scenes(root: Path):
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        bands = find_bands(d)
        if not bands:
            continue
        metas = sorted(d.glob("*.meta"))
        if not metas:
            metas = [f for f in sorted(d.glob("*.txt"))
                     if "ACC_REP" not in f.name.upper()]
        yield d, bands, (metas[0] if metas else None)


def scene_id(folder: Path, meta: dict, fallback: str) -> str:
    for k in ("OTSProductID", "ProductID"):
        if meta.get(k):
            return meta[k]
    return folder.name or fallback


# ==========================================================================
# geometry 
# ==========================================================================
def scan_rows(band_path: Path, fill, block: int = 512):
    with rasterio.open(band_path) as src:
        h, w = src.height, src.width
        first = np.full(h, -1, dtype=np.int64)
        last = np.full(h, -1, dtype=np.int64)
        is_nan = isinstance(fill, float) and np.isnan(fill)
        for r0 in range(0, h, block):
            n = min(block, h - r0)
            a = src.read(1, window=Window(0, r0, w, n))
            valid = ~np.isnan(a) if is_nan else (a != fill)
            any_valid = valid.any(axis=1)
            f = valid.argmax(axis=1)
            l = w - 1 - valid[:, ::-1].argmax(axis=1)
            first[r0:r0 + n] = np.where(any_valid, f, -1)
            last[r0:r0 + n] = np.where(any_valid, l, -1)
    return first, last, h, w


def largest_row_block(runs: np.ndarray):
    h = len(runs)
    best = (0, 0, 0, 0)
    stack = []
    for i in range(h + 1):
        cur = int(runs[i]) if i < h else 0
        start = i
        while stack and stack[-1][1] > cur:
            idx, ht = stack.pop()
            area = ht * (i - idx)
            if area > best[0]:
                best = (area, idx, i, ht)
            start = idx
        stack.append((start, cur))
    return best[1], best[2], best[3]


def fit_shear(starts: np.ndarray, r0: int, r1: int):
    rows = np.arange(r1 - r0, dtype=float)
    m, s0 = np.polyfit(rows, starts[r0:r1].astype(float), 1)
    dev = np.abs(starts[r0:r1] - (s0 + m * rows))
    return float(m), float(s0), float(dev.max()), float(dev.mean())


# ==========================================================================
# writers
# ==========================================================================
def _profile_single(ref, height, width, transform, nodata):
    dt = np.dtype(ref.dtypes[0])
    prof = dict(
        driver="GTiff", height=int(height), width=int(width), count=1,
        dtype=ref.dtypes[0], crs=ref.crs, transform=transform,
        tiled=True, blockxsize=512, blockysize=512,
        compress="DEFLATE", predictor=3 if dt.kind == "f" else 2,
        BIGTIFF="IF_SAFER", num_threads="ALL_CPUS",
    )
    if nodata is not None:
        prof["nodata"] = nodata
    return prof


def write_shift_split(band_paths, out_paths, starts, r0, r1, width,
                      out_transform, fill, block=512):
    h_out = r1 - r0
    for band_path, out_path in zip(band_paths, out_paths):
        src = rasterio.open(band_path)
        try:
            prof = _profile_single(src, h_out, width, out_transform, fill)
            with rasterio.open(out_path, "w", **prof) as dst:
                for off in range(0, h_out, block):
                    n = min(block, h_out - off)
                    s = starts[r0 + off: r0 + off + n]
                    lo, hi = int(s.min()), min(int(s.max()) + width, src.width)
                    win = Window(lo, r0 + off, hi - lo, n)
                    chunk = src.read(1, window=win)
                    buf = np.empty((n, width), dtype=prof["dtype"])
                    for k in range(n):
                        a = int(s[k]) - lo
                        buf[k] = chunk[k, a:a + width]
                    dst.write(buf, 1, window=Window(0, off, width, n))
                dst.descriptions = (band_path.stem,)
        finally:
            src.close()


def write_warp_split(band_paths, out_paths, out_transform, height, width,
                     fill, resampling="cubic"):
    for band_path, out_path in zip(band_paths, out_paths):
        with rasterio.open(band_path) as src:
            prof = _profile_single(src, height, width, out_transform, fill)
            buf = np.full((height, width), fill if fill is not None else 0,
                          dtype=prof["dtype"])
            reproject(
                source=rasterio.band(src, 1), destination=buf,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=out_transform, dst_crs=src.crs,
                src_nodata=fill, dst_nodata=fill,
                resampling=getattr(Resampling, resampling), num_threads=2,
            )
            with rasterio.open(out_path, "w", **prof) as dst:
                dst.write(buf, 1)
                dst.descriptions = (band_path.stem,)


# ==========================================================================
# JPG preview 
# ==========================================================================
def tif_to_jpg(tif_path: Path, jpg_path: Path, quality: int = 90):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        h, w = src.height, src.width
    mask = (arr != nodata) if nodata is not None else np.ones_like(arr, dtype=bool)
    valid = arr[mask]
    if valid.size == 0:
        LOG.warning("      %s: no valid pixels, skipping JPG preview", tif_path.name)
        return None
    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    scaled[~mask] = 0
    Image.fromarray(scaled, mode="L").save(jpg_path, "JPEG", quality=quality)
    return jpg_path, w, h


# ==========================================================================
# per-scene 
# ==========================================================================
def rectangle_from_scene(folder, bands, meta_path, out_dir, a, source_zip=""):
    t0 = time.time()
    meta = parse_meta(meta_path) if meta_path else {}
    sid = scene_id(folder, meta, "scene")

    # Route each band to its specific subdirectory (BAND2, BAND3, etc.)
    out_paths = []
    for b in bands:
        b_num = band_number(b)
        b_folder = Path(out_dir) / f"BAND{b_num}"
        b_folder.mkdir(parents=True, exist_ok=True)
        out_paths.append(b_folder / f"{sid}_rect.tif")

    tmp_paths = [p.with_name(f"{p.name}.partial.{os.getpid()}") for p in out_paths]

    if a.resume and all(p.exists() and p.stat().st_size > 0 for p in out_paths):
        LOG.info("SKIP  %s (outputs exist)", sid)
        return None

    with rasterio.open(bands[0]) as s:
        if s.crs is None:
            LOG.error("FAIL  %s: %s has no CRS", sid, bands[0].name)
            return None
        fill = a.fill if a.fill is not None else (s.nodata if s.nodata is not None else 0)
        in_transform, in_w, in_h = s.transform, s.width, s.height

    first, last, h, w = scan_rows(bands[0], fill, a.block)
    runs = np.where(first >= 0, last - first + 1 - 2 * a.inset, 0).clip(0)
    if runs.max() <= 0:
        LOG.error("FAIL  %s: no valid pixels (try --fill)", sid)
        return None

    r0, r1, width = largest_row_block(runs)
    if width <= 0 or r1 <= r0:
        LOG.error("FAIL  %s: could not fit a rectangle", sid)
        return None

    starts = (first + a.inset).clip(0, w - width)
    m, s0, dev_max, dev_mean = fit_shear(starts, r0, r1)

    out_transform = in_transform * Affine(1.0, m, s0, 0.0, 1.0, float(r0))

    for tp in tmp_paths:
        tp.unlink(missing_ok=True)
    try:
        if a.method == "shift":
            write_shift_split(bands, tmp_paths, starts, r0, r1, width,
                               out_transform, fill, a.block)
        else:
            write_warp_split(bands, tmp_paths, out_transform, r1 - r0, width,
                              fill, a.resampling)
        for tp, op in zip(tmp_paths, out_paths):
            os.replace(tp, op)
    except BaseException:
        for tp in tmp_paths:
            tp.unlink(missing_ok=True)
        raise

    jpg_paths = []
    jpg_dims = []
    if not a.no_jpg:
        for op in out_paths:
            jp = op.with_suffix(".jpg")
            try:
                result = tif_to_jpg(op, jp, a.jpg_quality)
                if result:
                    jpg_paths.append(jp)
                    jpg_dims.append((result[1], result[2]))
            except Exception:
                LOG.error("FAIL  %s: JPG conversion failed\n%s",
                          op.name, traceback.format_exc())

    valid_in = int(runs[runs > 0].sum())
    kept = width * (r1 - r0)
    col_step = abs(in_transform.a)
    row_step = (abs(in_transform.e) ** 2 + (m * col_step) ** 2) ** 0.5
    total_bytes = sum(p.stat().st_size for p in out_paths)
    secs = time.time() - t0

    LOG.info(
        "OK    %s | in %dx%d valid %.1f%% | out %dx%d x%d files (+%d jpg) "
        "%.2f%% of valid kept | shear %+.5f (%.2f deg) | %s | %.0f MB total | %.0fs",
        sid, in_w, in_h, 100 * valid_in / (in_w * in_h), width, r1 - r0,
        len(out_paths), len(jpg_paths), 100 * kept / valid_in, m,
        np.degrees(np.arctan(-m)), a.method, total_bytes / (1 << 20), secs,
    )
    LOG.info(
        "      %s | input tif dims: %dx%d (w x h) | output tif dims: %dx%d "
        "(w x h) per band | output jpg dims: %s (w x h) per band",
        sid, in_w, in_h, width, r1 - r0,
        ", ".join(f"{jw}x{jh}" for jw, jh in jpg_dims) if jpg_dims else "n/a (--no-jpg)",
    )
    if a.method == "shift" and dev_max > 1.5:
        LOG.warning("      %s: edge not cleanly linear (max dev %.2f px). "
                    "Check --inset, or use --method warp.", sid, dev_max)

    return {
        "scene_id": sid,
        "out_files": ";".join(p.name for p in out_paths),
        "jpg_files": ";".join(p.name for p in jpg_paths),
        "source_zip": source_zip,
        "satellite": meta.get("SatID", ""), "sensor": meta.get("Sensor", ""),
        "date": meta.get("DateOfPass", ""), "path": meta.get("Path", ""),
        "row": meta.get("Row", ""), "bands": meta.get("BandNumbers", ""),
        "epsg": rasterio.open(out_paths[0]).crs.to_epsg(), "method": a.method,
        "in_width": in_w, "in_height": in_h,
        "out_width": width, "out_height": r1 - r0,
        "jpg_width": jpg_dims[0][0] if jpg_dims else "",
        "jpg_height": jpg_dims[0][1] if jpg_dims else "",
        "valid_pct_of_grid": round(100 * valid_in / (in_w * in_h), 2),
        "kept_pct_of_valid": round(100 * kept / valid_in, 3),
        "shear_col_per_row": round(m, 6),
        "shear_deg": round(float(np.degrees(np.arctan(-m))), 3),
        "col_step_m": round(col_step, 3), "row_step_m": round(row_step, 3),
        "dev_max_px": round(dev_max, 3), "dev_mean_px": round(dev_mean, 3),
        "ulx": round(out_transform.c, 3), "uly": round(out_transform.f, 3),
        "out_bytes": total_bytes, "seconds": round(secs, 1),
    }


def process_input(src_str, out_str, a):
    src, out_dir = Path(src_str), Path(out_str)
    rows, scratch = [], None
    try:
        if src.suffix.lower() == ".zip":
            scratch = Path(tempfile.mkdtemp(prefix="bhrect_", dir=a.scratch))
            try:
                with zipfile.ZipFile(src) as z:
                    z.extractall(scratch)
            except zipfile.BadZipFile:
                LOG.error("FAIL  %s: corrupt or truncated zip", src.name)
                return []
            root, tag = scratch, src.name
        else:
            root, tag = src, ""

        found = False
        for folder, bands, meta_path in find_scenes(root):
            found = True
            try:
                r = rectangle_from_scene(folder, bands, meta_path, out_dir, a, tag)
                if r:
                    rows.append(r)
            except Exception:
                LOG.error("FAIL  %s\n%s", folder.name, traceback.format_exc())
        if not found:
            LOG.warning("SKIP  %s: no BAND*.tif inside", src.name)
        elif rows and a.delete_zip_after and src.suffix.lower() == ".zip":
            src.unlink()
            LOG.info("      deleted %s", src.name)
        return rows
    finally:
        if scratch and not a.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def _worker(args):
    src, out, a = args
    setup_logging(a.log, a.quiet)
    return process_input(src, out, a)


# ==========================================================================
# preflight 
# ==========================================================================
def probe_sizes(zips, a):
    for z in zips:
        if Path(z).suffix.lower() != ".zip":
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                infos = [i for i in zf.infolist()
                         if i.filename.lower().endswith((".tif", ".tiff"))
                         and re.search(r"(?:BAND|B)[_-]?\d",
                                       Path(i.filename).name, re.IGNORECASE)]
                if infos:
                    return (sum(i.file_size for i in infos),
                            Path(z).stat().st_size, len(infos))
        except zipfile.BadZipFile:
            continue
    return None


def preflight(zips, out_dir, a):
    probe = probe_sizes(zips, a)
    n = len(zips)
    LOG.info("inputs: %d", n)
    if not probe:
        LOG.warning("could not size the job (no readable zip) - skipping disk check")
        return True

    unzipped, zipped, nbands = probe
    est_out = unzipped * 0.55
    need_scratch = unzipped * max(1, a.workers)
    need_out = est_out * n

    LOG.info("per scene: %d band(s), %.2f GiB unzipped, %.2f GiB zip, "
             "~%.2f GiB output (est.)",
             nbands, unzipped / GB, zipped / GB, est_out / GB)
    LOG.info("job total: ~%.1f GiB of output, ~%.1f GiB scratch live "
             "at --workers %d", need_out / GB, need_scratch / GB, a.workers)

    ok = True
    for label, path, need in (("output", out_dir, need_out),
                              ("scratch", a.scratch or tempfile.gettempdir(),
                               need_scratch)):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free
        LOG.info("%-7s %s: %.1f GiB free, ~%.1f GiB needed",
                 label, p, free / GB, need / GB)
        if free < need * 1.10:
            LOG.error("NOT ENOUGH SPACE for %s. Free up disk, lower --workers, "
                      "or point --scratch elsewhere.", label)
            ok = False
    return ok


def setup_logging(log_path, quiet):
    root = logging.getLogger("rect")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    if not quiet:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        root.addHandler(h)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)


# ==========================================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Updated defaults for the pipeline paths
    p.add_argument("inp", nargs="?", default="/workspace/Cloudy-with-a-chance-of-kofta/Bhoonidhi-Data/data", 
                   help=".zip, directory of .zip files, or unzipped scene folder")
    p.add_argument("out", nargs="?", default="/workspace/Cloudy-with-a-chance-of-kofta/Bhoonidhi-Data/data_processed/target_A", 
                   help="output directory base (BAND2, BAND3, etc. will be created inside here)")

    g = p.add_argument_group("geometry")
    g.add_argument("--method", choices=["shift", "warp"], default="shift",
                   help="shift = lossless integer row shift (default); "
                        "warp = exact affine resample")
    g.add_argument("--resampling", default="cubic",
                   choices=["nearest", "bilinear", "cubic", "lanczos"],
                   help="kernel for --method warp")
    g.add_argument("--inset", type=int, default=1,
                   help="drop N columns at each row edge to avoid partially "
                        "filled boundary pixels (default 1)")
    g.add_argument("--fill", type=float, default=None,
                   help="pad value in the source (default: file nodata, else 0)")

    j = p.add_argument_group("jpg preview")
    j.add_argument("--no-jpg", action="store_true",
                   help="skip auto-generating a .jpg preview for each band tif")
    j.add_argument("--jpg-quality", type=int, default=90,
                   help="JPEG quality 1-95 (default 90)")

    r = p.add_argument_group("run control")
    r.add_argument("--workers", type=int, default=1,
                   help="zips in parallel; EACH unzips a whole scene, so scratch "
                        "scales linearly. Start at 3.")
    r.add_argument("--scratch", default=None,
                   help="where to unzip. Use fast local disk, not a network volume.")
    r.add_argument("--block", type=int, default=512, help="row block size for I/O")
    r.add_argument("--gdal-cache", type=int, default=256,
                   help="GDAL block cache in MB per worker (default 256)")
    r.add_argument("--resume", action="store_true",
                   help="skip scenes whose output already exists")
    r.add_argument("--keep-scratch", action="store_true")
    r.add_argument("--delete-zip-after", action="store_true",
                   help="DESTRUCTIVE: delete each zip once its scenes succeed")
    r.add_argument("--dry-run", action="store_true",
                   help="size the job and check disk, then exit")
    r.add_argument("--skip-disk-check", action="store_true")
    r.add_argument("--manifest", default="manifest.csv")
    r.add_argument("--log", default=None, help="also write logs to this file")
    r.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    setup_logging(a.log, a.quiet)
    os.environ.setdefault("GDAL_CACHEMAX", str(a.gdal_cache))

    src, out_dir = Path(a.inp), Path(a.out)
    if src.is_dir() and not find_bands(src) and list(src.rglob("*.zip")):
        inputs = sorted(src.rglob("*.zip"))
    else:
        inputs = [src]
    if not inputs or (len(inputs) == 1 and not inputs[0].exists()):
        LOG.error("nothing to process at %s", src)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    if a.scratch:
        Path(a.scratch).mkdir(parents=True, exist_ok=True)

    LOG.info("method=%s  workers=%d  resume=%s", a.method, a.workers, a.resume)
    ok = preflight(inputs, out_dir, a)
    if a.dry_run:
        LOG.info("dry run - nothing written")
        return 0 if ok else 1
    if not ok and not a.skip_disk_check:
        LOG.error("aborting. Re-run with --skip-disk-check to override.")
        return 1

    man_path = out_dir / a.manifest
    new = not man_path.exists()
    man = man_path.open("a", newline="")
    writer = csv.DictWriter(man, fieldnames=MANIFEST_COLS, extrasaction="ignore")
    if new:
        writer.writeheader()

    jobs = [(str(i), str(out_dir), a) for i in inputs]
    t0, done, total_rows = time.time(), 0, 0

    def record(rows):
        nonlocal done, total_rows
        done += 1
        for row in rows:
            writer.writerow(row)
            total_rows += 1
        man.flush()
        el = time.time() - t0
        eta = (el / done) * (len(jobs) - done)
        LOG.info("--- %d/%d inputs, %d scene(s) written, %.0f min elapsed, "
                 "~%.0f min left", done, len(jobs), total_rows, el / 60, eta / 60)

    try:
        if a.workers > 1:
            with ProcessPoolExecutor(a.workers) as ex:
                futs = [ex.submit(_worker, j) for j in jobs]
                for f in as_completed(futs):
                    record(f.result())
        else:
            for j in jobs:
                record(_worker(j))
    except KeyboardInterrupt:
        LOG.warning("interrupted. Partial outputs were discarded; "
                    "re-run with --resume to continue.")
        return 130
    finally:
        man.close()

    LOG.info("done: %d scene(s) in %.1f min. Manifest: %s",
             total_rows, (time.time() - t0) / 60, man_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())