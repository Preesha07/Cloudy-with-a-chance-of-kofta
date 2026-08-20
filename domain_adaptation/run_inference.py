#!/usr/bin/env python3
"""
Domain-adaptive cloud removal inference pipeline.

Reads per-band scaled JPEG images (BAND2/BAND3/BAND4 layout), merges them into
3-channel RGB patches on the fly, runs the student model with the chosen
adaptation method, stitches per-tile outputs back to full-scene images, and
saves all intermediate outputs.

Output layout (one folder per scene):
    <out_root>/<method>/<scene_id>/
        final.jpg          — full stitched output after domain restoration
        final_para.jpg     — final.jpg re-sheared to pushbroom shape
        fake_b.jpg         — raw model composite (before domain restoration)
        fake_c.jpg         — netH hallucinated SAR-like image
        g_b.jpg            — raw netG decoder output (before attention blending)
        attention.jpg      — att_A cloud-probability heatmap
        patches/<name>/*.jpg — per-tile 256x256 outputs (only with --save_patches),
                                one dir per output type, filenames "y{y}_x{x}.jpg"

Usage:
    python run_inference.py \\
        --method        dataset_norm \\
        --model_variant coral \\
        --scaled_root   Bhoonidhi-Data/data_scaled \\
        --weights       <ckpt_root>/sen12mscr_student_coral_adapted/latest_net_G.pth \\
        --out_root      Bhoonidhi-Data/outputs_adabn \\
        --rect_root     Bhoonidhi-Data/data_rect \\
        --adabn --adabn_batches 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Allow running from anywhere in the repo
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from domain_adaptation.pipeline.io import (
    LazyPatchDataset,
    tensor_to_uint8,
    attn_to_uint8,
    tensor_to_hwc,
    attn_to_hwc,
)
from domain_adaptation.pipeline.model import load_student_model, apply_ttbn, run_adabn, try_compile
from domain_adaptation.pipeline.stitch import (
    Stitcher,
    read_manifest,
    get_shear_params,
    apply_pushbroom_shear,
    save_image,
)
from domain_adaptation.methods import get_adapter

Image.MAX_IMAGE_PIXELS = None
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Reference image loading
# ---------------------------------------------------------------------------

def load_ref_tensor(ref_image_path: str | None, device: torch.device) -> torch.Tensor | None:
    """Load a SEN12MS-CR reference image as a [1, 3, 256, 256] tensor in [-1, 1]."""
    if ref_image_path is None:
        return None
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = Image.open(ref_image_path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Per-scene inference
# ---------------------------------------------------------------------------

def _canvas_key(name: str) -> tuple[str, int]:
    """Return (canvas_name, num_channels) for each output type."""
    return {"final": 3, "fake_b": 3, "fake_c": 3, "g_b": 3, "attention": 1}[name]


def infer_scene(
    scene_id: str,
    scene_band_dir: Path,
    netG, netA, netH,
    device: torch.device,
    adapter,
    ref_tensor: torch.Tensor | None,
    out_dir: Path,
    manifest_row: dict | None,
    args: argparse.Namespace,
) -> None:
    """Run full inference + stitching for one scene."""
    t0 = time.time()

    # --- Dataset -----------------------------------------------------------
    band2 = scene_band_dir / "BAND2"
    band3 = scene_band_dir / "BAND3"
    band4 = scene_band_dir / "BAND4"

    dataset = LazyPatchDataset(band2, band3, band4, patch_size=256, stride=128,
                               stretch_input=adapter.stretch_input,
                               scene_transform=adapter.scene_transform)
    orig_h, orig_w = dataset.orig_h, dataset.orig_w
    n_total = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=args.num_workers > 0,
        drop_last=adapter.drop_last,
    )

    # --- AdaBN (optional) ---------------------------------------------------
    # Two independent switches can request it: the adapter itself (e.g.
    # cloud_quantile always wants it) or the top-level --adabn CLI flag, which
    # applies regardless of adapter and is the one-click revert (just omit it).
    if getattr(adapter, 'adabn', False) or getattr(args, 'adabn', False):
        n_batches = getattr(args, 'adabn_batches', None) or getattr(adapter, 'adabn_batches', 50)
        print(f"  AdaBN: {n_batches} batches …")
        run_adabn(netG, netA, netH, loader, device, n_batches=n_batches)

    # --- Per-patch output dirs (optional) -----------------------------------
    patch_dirs = None
    if getattr(args, 'save_patches', False):
        patch_dirs = {name: out_dir / "patches" / name for name in
                      ("final", "fake_b", "fake_c", "g_b", "attention")}
        for d in patch_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    # --- Stitch canvases ---------------------------------------------------
    patch_size, stride = 256, 128
    specs = {"final": 3, "fake_b": 3, "fake_c": 3, "g_b": 3, "attention": 1}

    stitch_device = args.stitch_device
    if stitch_device == "auto":
        need = Stitcher.bytes_needed(specs, orig_h, orig_w, patch_size, stride)
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            # Leave headroom for the model's own activations.
            stitch_device = "cuda" if need < free - 3 * (1 << 30) else "cpu"
        else:
            stitch_device = "cpu"
        print(f"  stitch on {stitch_device} ({need / (1 << 30):.1f} GiB of canvases)")

    stitcher = Stitcher(specs, orig_h, orig_w, patch_size, stride, device=stitch_device)

    # --- Inference loop ----------------------------------------------------
    ref_batch = ref_tensor  # will be broadcast in loop
    n_done = 0

    with torch.no_grad():
        for tensors, ys, xs in loader:
            tensors = tensors.to(device, non_blocking=True)
            B = tensors.size(0)

            if ref_batch is not None:
                rb = ref_batch.expand(B, -1, -1, -1)
            else:
                rb = None

            adapted = adapter.adapt_input(tensors, rb)

            with torch.amp.autocast("cuda"):
                fake_c = netH(adapted)
                att_A  = netA(adapted)
                g_b    = netG(torch.cat([adapted, fake_c], dim=1))
                fake_b = g_b * att_A + adapted * (1.0 - att_A)

            final = adapter.restore_output(fake_b, tensors)

            # Stay on-device: [B, H, W, C] float 0-255, no host round-trip.
            patches = {
                "final":     tensor_to_hwc(final),
                "fake_b":    tensor_to_hwc(fake_b),
                "fake_c":    tensor_to_hwc(fake_c),
                "g_b":       tensor_to_hwc(g_b),
                "attention": attn_to_hwc(att_A),
            }
            if stitch_device != str(device):
                patches = {k: v.to(stitch_device) for k, v in patches.items()}

            ys_np = ys.numpy()
            xs_np = xs.numpy()
            stitcher.add(patches, ys_np, xs_np)

            if patch_dirs is not None:
                np_final  = tensor_to_uint8(final)
                np_fake_b = tensor_to_uint8(fake_b)
                np_fake_c = tensor_to_uint8(fake_c)
                np_g_b    = tensor_to_uint8(g_b)
                np_att    = attn_to_uint8(att_A)
                for i in range(B):
                    y, x = int(ys_np[i]), int(xs_np[i])
                    save_image(np_final[i],  patch_dirs["final"]     / f"y{y}_x{x}.jpg", stretch=False)
                    save_image(np_fake_b[i], patch_dirs["fake_b"]    / f"y{y}_x{x}.jpg", stretch=False)
                    save_image(np_fake_c[i], patch_dirs["fake_c"]    / f"y{y}_x{x}.jpg", stretch=False)
                    save_image(np_g_b[i],    patch_dirs["g_b"]       / f"y{y}_x{x}.jpg", stretch=False)
                    save_image(np_att[i],    patch_dirs["attention"] / f"y{y}_x{x}.jpg", stretch=False)

            n_done += B
            print(f"\r  {n_done}/{n_total} tiles", end="", flush=True)

    print(f"\n  stitching…")

    # --- Bias correction ----------------------------------------------------
    # The InstanceNorm layers in this student use accumulated SEN12MS-CR running
    # statistics at eval time (track_running_stats=True in get_norm_layer), which
    # applies a roughly constant per-channel DC offset on out-of-domain LISS-4 input
    # — the model output ends up systematically darker than it should be, with the
    # image structure otherwise intact ("info is there, just offset"). Correct it by
    # matching the raw output's clear-region (non-cloud) mean to the adapted input
    # scene's own clear-region mean, per channel, and adding that constant back.
    fake_b_raw = stitcher.finalize("fake_b").astype(np.float32)
    att_raw    = stitcher.finalize("attention").astype(np.float32)
    adapted_in = dataset._arr[:orig_h, :orig_w, :].astype(np.float32)  # scene_transform'd input, model channel order

    clear_mask = att_raw[:, :, 0] < 0.3
    if clear_mask.sum() < 1000:
        clear_mask = np.ones(att_raw.shape[:2], dtype=bool)  # too little clear area — use whole scene

    channel_bias = (
        adapted_in[clear_mask].mean(axis=0) - fake_b_raw[clear_mask].mean(axis=0)
    )
    print(f"  bias correction (R,G,NIR): {channel_bias.round(1).tolist()}  "
          f"(clear pixels: {int(clear_mask.sum())}/{clear_mask.size})")

    # --- Finalize and save ------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in specs:
        arr = stitcher.finalize(name)
        if name in ("final", "fake_b", "g_b") and arr.shape[2] == 3:
            arr = np.clip(arr.astype(np.float32) + channel_bias, 0, 255).astype(np.uint8)
        # final output: remap channels from model order [R, G, NIR] → [NIR, R, G]
        # so NIR appears in the red display channel (false-colour composite).
        if name == "final" and arr.shape[2] == 3:
            arr = arr[:, :, [2, 0, 1]]
        save_image(arr, out_dir / f"{name}.jpg", stretch=False)
        print(f"  saved {name}.jpg")

    # --- Parallelogram output ---------------------------------------------
    if manifest_row is not None:
        m, s0, r0, rect_h, in_h, in_w = get_shear_params(manifest_row)
        gsd_scale = orig_h / rect_h
        in_h_s = round(in_h * gsd_scale)
        in_w_s = round(in_w * gsd_scale)
        s0_s   = round(s0   * gsd_scale)
        final_arr = stitcher.finalize("final")
        final_arr = np.clip(final_arr.astype(np.float32) + channel_bias, 0, 255).astype(np.uint8)
        # same NIR-in-red remap before shearing
        final_arr = final_arr[:, :, [2, 0, 1]]
        para = apply_pushbroom_shear(final_arr, m, s0_s, r0, in_h_s, in_w_s)
        save_image(para, out_dir / "final_para.jpg", stretch=False)
        print("  saved final_para.jpg")
    else:
        print("  (no manifest → parallelogram output skipped)")

    elapsed = time.time() - t0
    print(f"  done in {elapsed:.0f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--method", default="dataset_norm", choices=["dataset_norm"],
                   help="Adaptation method (only dataset_norm/TTBN is kept in this tree)")
    p.add_argument("--scaled_root", required=True,
                   help="Root dir with per-scene subdirs, each containing BAND2/BAND3/BAND4 scaled JPEGs")
    p.add_argument("--weights", required=True,
                   help="Path to student net_G weights (e.g. .../latest_net_G.pth)")
    p.add_argument("--out_root", required=True,
                   help="Root for outputs; per-scene dirs land at <out_root>/<method>/<scene_id>/")
    p.add_argument("--rect_root", default=None,
                   help="Root dir with per-scene manifest.csv files (data_rect/). "
                        "Required for parallelogram output.")
    p.add_argument("--model_variant", default="coral", choices=["coral"],
                   help="Student model class (only the CORAL-adapted student is kept here)")
    p.add_argument("--scene", default=None,
                   help="Process only this scene_id (default: all scenes found in scaled_root)")
    p.add_argument("--stitch_device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="where to accumulate the stitch canvases. 'auto' uses the "
                        "GPU when the canvases fit in free VRAM with 3 GiB spare, "
                        "else CPU.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_patches", action="store_true",
                   help="Also save each 256x256 model output tile to "
                        "<out_dir>/patches/<name>/y{y}_x{x}.jpg, before stitching/"
                        "bias-correction/channel-remap.")
    p.add_argument("--adabn", action="store_true",
                   help="Warm up BatchNorm/InstanceNorm running stats on LISS-4 batches "
                        "before inference (AdaBN), regardless of adapter. One-click revert: "
                        "just omit this flag.")
    p.add_argument("--adabn_batches", type=int, default=None,
                   help="Number of warmup batches for --adabn (default: adapter's own, or 50).")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = get_adapter(args.method)

    # Load model
    netG, netA, netH, device = load_student_model(args.weights, args.model_variant)
    for net in (netG, netA, netH):
        apply_ttbn(net)
    netG, netA, netH = try_compile(netG, netA, netH)

    ref_tensor = None   # dataset_norm/TTBN needs no reference image

    # Load all manifests (for parallelogram output)
    manifests: dict[str, dict] = {}
    if args.rect_root:
        rect_root = Path(args.rect_root)
        for manifest_path in rect_root.rglob("manifest.csv"):
            scene_manifests = read_manifest(manifest_path)
            manifests.update(scene_manifests)

    # Discover scenes
    scaled_root = Path(args.scaled_root)
    if args.scene:
        scene_dirs = [scaled_root / args.scene]
    else:
        scene_dirs = [d for d in sorted(scaled_root.iterdir()) if d.is_dir()]

    out_root = Path(args.out_root) / adapter.name
    print(f"Method: {adapter.name}  |  {len(scene_dirs)} scene(s)  |  batch={args.batch_size}")

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"\n=== {scene_id} ===")

        band2 = scene_dir / "BAND2"
        if not band2.exists() or not list(band2.glob("*.jpg")):
            print("  skipping — no BAND2/*.jpg found")
            continue

        infer_scene(
            scene_id      = scene_id,
            scene_band_dir= scene_dir,
            netG=netG, netA=netA, netH=netH,
            device        = device,
            adapter       = adapter,
            ref_tensor    = ref_tensor,
            out_dir       = out_root / scene_id,
            manifest_row  = manifests.get(scene_id),
            args          = args,
        )


if __name__ == "__main__":
    main()
