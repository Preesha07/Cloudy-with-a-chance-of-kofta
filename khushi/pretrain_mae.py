"""Short MAE pretraining run to verify the ViT backbone learns on Sentinel data."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import masked_psnr, ssim
from sen12mscr_dataset import SEN12MSCROptical, denormalize
from vit_pretrain import SIZE_PRESETS, MAEPretrainer, config_for_size


# --------------------------------------------------------------------------- #
# Muon optimizer (2D hidden weight matrices in the transformer blocks only) —
# ported from vandita/trying.py so this driver also benefits from it.
# AdamW keeps handling everything else: patch/pos embeddings, norms, biases,
# the decoder's io heads.
# --------------------------------------------------------------------------- #
def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize a 2D gradient/momentum matrix via Newton-Schulz iteration."""
    assert g.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        A = x @ x.T
        B = b * A + c * A @ A
        x = a * x + B @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: SGD-momentum + Newton-Schulz orthogonalization, for 2D matrices."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon params must be 2D weight matrices"
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = _zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if wd:
                    p.mul_(1 - lr * wd)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)


def build_param_groups(model: nn.Module):
    """Split params: 2D weight matrices inside transformer blocks -> Muon;
    everything else -> AdamW (with/without weight decay by ndim)."""
    muon_params, adamw_decay, adamw_no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        in_block = ".blocks." in name
        if p.ndim == 2 and in_block:
            muon_params.append(p)
        elif p.ndim >= 2:
            adamw_decay.append(p)
        else:
            adamw_no_decay.append(p)
    return muon_params, adamw_decay, adamw_no_decay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../SEN12MSCR/ROIs1158_spring_s2")
    p.add_argument("--pattern", default="**/*.tif")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--model-size", choices=sorted(SIZE_PRESETS), default="small",
                   help="encoder/decoder shape preset. 'small' (~22M encoder) is the "
                        "default: SEN12MS-CR's spring ROI is ~29K tiles, far too few "
                        "to converge a Base encoder in a realistic step budget")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation; effective batch = batch-size * accum-steps")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0,
                   help="cap on optimizer steps (not micro-batches); 0 = use epochs")
    p.add_argument("--lr", type=float, default=1.5e-4,
                   help="reference LR at effective batch 256 (MAE convention); "
                        "scaled by effective_batch/256 unless --no-lr-scaling")
    p.add_argument("--no-lr-scaling", action="store_true",
                   help="use --lr verbatim instead of the linear batch-size scaling rule")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="warmup in optimizer steps; 0 = auto (one epoch)")
    p.add_argument("--recon-loss", choices=["l2", "l1", "charbonnier"], default="l2",
                   help="pixel reconstruction objective; l1/charbonnier sharpen output")
    p.add_argument("--no-muon", action="store_true",
                   help="disable Muon; use plain AdamW for every parameter (old behavior)")
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="reference Muon LR at effective batch 256, only for the "
                        "2D in-block weight matrices; scaled the same way as --lr")
    p.add_argument("--cmmd-weight", type=float, default=0.0,
                   help="weight of the CMMD-style embedding-space MMD term "
                        "(encoder features of real vs. masked-patch reconstruction); "
                        "0 = off (default, no extra encoder forward passes)")
    p.add_argument("--cmmd-sigma", type=float, default=0.0,
                   help="RBF bandwidth for the CMMD term; 0 = auto (median-distance heuristic)")
    p.add_argument("--norm-pix-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="per-patch mean/std normalize reconstruction targets before the "
                        "loss (masked patches only actually contribute to the loss; "
                        "visible patches' normalized values are computed but discarded) "
                        "— keeps the ViT from just learning the per-patch mean pixel")
    p.add_argument("--augment", action="store_true",
                   help="random flips + rot90 on tiles (label-preserving for satellite)")
    p.add_argument("--photometric", type=float, default=0.0,
                   help="radiometric jitter strength (0 = off, 1.0 = +-25%% per-channel "
                        "gain, +-0.02 offset, gamma in [1/1.3, 1.3]). Guards against "
                        "the encoder overfitting Sentinel-2's exact calibration when "
                        "the student will be fed LISS-4")
    p.add_argument("--rrc", type=float, nargs=2, metavar=("MIN", "MAX"), default=None,
                   help="random-resized-crop area fraction, e.g. --rrc 0.3 1.0. Crops "
                        "at a random aspect ratio and resizes back to --img-size, so "
                        "the encoder sees varied ground sample distances (LISS-4 is "
                        "5.8m vs Sentinel-2's 10m)")
    p.add_argument("--no-normalize", action="store_true",
                   help="feed raw reflectance in [0,1] instead of per-channel "
                        "standardized input (the pre-2026-07 behavior)")
    p.add_argument("--min-dynamic-range", type=float, default=0.03,
                   help="drop tiles whose largest per-channel 98th-2nd reflectance "
                        "percentile span is below this (0 = keep all). Measured over "
                        "500 tiles of ROIs1158_spring: median 0.177, p5 0.060, and "
                        "blank water/shadow tiles near 0.005 — the default drops "
                        "~3.6%%. Under norm_pix_loss those blank tiles train the "
                        "model to reproduce sensor artifact and flatter val metrics. "
                        "Costs a one-time cached scan of the dataset")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="fraction of tiles held out for evaluation; 0 = no val split")
    p.add_argument("--eval-every", type=int, default=500,
                   help="run validation every N optimizer steps (0 = off)")
    p.add_argument("--eval-batches", type=int, default=16,
                   help="number of val batches per evaluation; 0 = the whole split")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default="mae_backbone.pt")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save a resumable checkpoint every N optimizer steps (0 = off); "
                        "written next to --out as <out>.ckpt")
    p.add_argument("--resume", default="",
                   help="path to a <out>.ckpt to resume from (restores model, optimizer, "
                        "scaler, and step counter)")
    p.add_argument("--log-file", default="train_log.txt",
                   help="append every log line here as well as stdout, so a run "
                        "survives the terminal it was launched from")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Logging: everything printed also lands in --log-file, flushed per line so a
# killed run keeps its history.
# --------------------------------------------------------------------------- #
_LOG_FH = None


def log(msg: str = "") -> None:
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(path, model, optim, muon, scaler, step, args) -> None:
    """Resumable checkpoint: full training state, written atomically.

    Distinct from the final --out artifact (which also carries the
    backbone-only state dict for the Phase-2 handoff). Written to a temp file
    then renamed so an interrupted save never corrupts the last good checkpoint.
    ``muon`` is ``None`` when running with ``--no-muon``.
    """
    tmp = f"{path}.tmp"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mae_config": vars(model.config),
            "optim_state_dict": optim.state_dict(),
            "muon_state_dict": muon.state_dict() if muon is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "step": step,
            "config": vars(args),
        },
        tmp,
    )
    Path(tmp).replace(path)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int, amp: bool,
             normalized: bool = True, seed: int = 1234):
    """Held-out reconstruction metrics.

    The mask is drawn from a private generator seeded identically on every call,
    so successive evaluations differ only by the model — without this, the val
    loss wobbles with mask luck and can't be compared across steps.

    Returns a dict with:
        loss      — the training objective on held-out tiles
        psnr_norm — PSNR on per-patch-normalized (unit-variance) targets; what
                    the model actually predicts under norm_pix_loss. This is the
                    honest number, and 0 dB is the do-nothing baseline of
                    predicting each patch's mean pixel.
        psnr_pix  — PSNR in reflectance space after restoring ground-truth
                    per-patch mean/std; optimistic, see metrics.py
        ssim      — SSIM of the composited image (visible GT + predicted masked)
    """
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)

    tot_loss, tot_psnr_n, tot_psnr_p, tot_ssim, n = 0.0, 0.0, 0.0, 0.0, 0
    for i, imgs in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            loss, pred, mask = model(imgs, generator=gen)
        pred = pred.float()

        target = model.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        std = (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5

        # Honest metric: compare in the normalized space the model predicts in.
        if model.config.norm_pix_loss:
            target_norm = (target - mean) / std
            pred_pix = pred * std + mean
        else:
            target_norm, pred_pix = target, pred
        tot_psnr_n += masked_psnr(pred, target_norm, mask, data_range=1.0)

        # Pixel-space metrics, in reflectance units so PSNR/SSIM are readable.
        mask_exp = mask.unsqueeze(-1)
        recon = model.unpatchify(mask_exp * pred_pix + (1 - mask_exp) * target)
        recon_r = (denormalize(recon) if normalized else recon).clamp(0.0, 1.0)
        imgs_r = (denormalize(imgs) if normalized else imgs).clamp(0.0, 1.0)
        tot_psnr_p += masked_psnr(
            model.patchify(recon_r), model.patchify(imgs_r), mask, data_range=1.0
        )
        tot_ssim += ssim(recon_r, imgs_r, data_range=1.0)

        tot_loss += loss.item()
        n += 1

    if was_training:
        model.train()
    if n == 0:
        return {}
    return {
        "loss": tot_loss / n,
        "psnr_norm": tot_psnr_n / n,
        "psnr_pix": tot_psnr_p / n,
        "ssim": tot_ssim / n,
    }


def main() -> None:
    global _LOG_FH
    args = parse_args()
    if args.log_file:
        _LOG_FH = open(args.log_file, "a")
        log("\n" + "=" * 78)
        log(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"args: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}")

    normalize = not args.no_normalize
    use_val = args.val_fraction > 0 and args.eval_every > 0
    ds = SEN12MSCROptical(
        args.data_root, pattern=args.pattern, img_size=args.img_size,
        augment=args.augment, normalize=normalize,
        photometric=args.photometric,
        rrc_scale=tuple(args.rrc) if args.rrc else None,
        split="train" if use_val else "all", val_fraction=args.val_fraction,
        min_dynamic_range=args.min_dynamic_range,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    val_loader = None
    if use_val:
        # No augmentation on val: the metric should track the model, not the
        # luck of which flip the held-out tile happened to get.
        val_ds = SEN12MSCROptical(
            args.data_root, pattern=args.pattern, img_size=args.img_size,
            augment=False, normalize=normalize,
            split="val", val_fraction=args.val_fraction,
            min_dynamic_range=args.min_dynamic_range, verbose=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=min(2, args.num_workers), pin_memory=(device == "cuda"),
            drop_last=True,
        )

    # Effective batch = micro-batch * accumulation. LR follows the MAE linear
    # scaling rule (reference LR is calibrated at effective batch 256), unless
    # explicitly disabled. Steps below count *optimizer* steps, not micro-batches.
    accum = max(1, args.accum_steps)
    eff_batch = args.batch_size * accum
    base_lr = args.lr if args.no_lr_scaling else args.lr * eff_batch / 256.0
    muon_base_lr = args.muon_lr if args.no_lr_scaling else args.muon_lr * eff_batch / 256.0
    steps_per_epoch = len(loader) // accum
    warmup = args.warmup_steps or steps_per_epoch  # default: one epoch of warmup
    total_steps = args.max_steps or args.epochs * steps_per_epoch
    log(f"tiles={len(ds)}  val_tiles={len(val_loader.dataset) if val_loader else 0}  "
          f"micro-batches/epoch={len(loader)}  opt-steps/epoch={steps_per_epoch}")
    log(f"eff_batch={eff_batch} (batch {args.batch_size} x accum {accum})  "
          f"base_lr={base_lr:.2e}  muon_lr={muon_base_lr:.2e} (muon={not args.no_muon})  "
          f"warmup={warmup}  total_steps={total_steps}  "
          f"recon={args.recon_loss}  norm_pix_loss={args.norm_pix_loss}  "
          f"cmmd_weight={args.cmmd_weight}  augment={args.augment}  "
          f"normalize={normalize}  photometric={args.photometric}  "
          f"rrc={tuple(args.rrc) if args.rrc else None}  "
          f"min_dyn_range={args.min_dynamic_range}")

    model = MAEPretrainer(
        config_for_size(
            args.model_size, in_channels=3, img_size=args.img_size,
            patch_size=args.patch_size,
            recon_loss=args.recon_loss, norm_pix_loss=args.norm_pix_loss,
            cmmd_weight=args.cmmd_weight, cmmd_sigma=args.cmmd_sigma or None,
        )
    ).to(device)
    enc_m = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    dec_m = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    log(f"model_size={args.model_size}  encoder={enc_m:.1f}M  decoder={dec_m:.1f}M  "
          f"mask_ratio={model.config.mask_ratio}")

    muon = None
    if args.no_muon:
        optim = torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
    else:
        muon_params, adamw_decay, adamw_no_decay = build_param_groups(model)
        optim = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": args.weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=base_lr, betas=(0.9, 0.95),
        )
        muon = Muon(muon_params, lr=muon_base_lr, weight_decay=args.weight_decay)
        log(f"muon params={sum(p.numel() for p in muon_params)/1e6:.1f}M  "
              f"adamw params={sum(p.numel() for p in adamw_decay + adamw_no_decay)/1e6:.1f}M")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optim_state_dict"])
        if muon is not None and ckpt.get("muon_state_dict") is not None:
            muon.load_state_dict(ckpt["muon_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt["step"]
        log(f"resumed from {args.resume} at step {start_step}/{total_steps}")

    ckpt_path = f"{args.out}.ckpt"
    best_path = f"{args.out}.best"
    best_psnr = float("-inf")  # selection metric: held-out psnr_norm
    step, t0 = start_step, time.time()
    model.train()
    optim.zero_grad(set_to_none=True)
    if muon is not None:
        muon.zero_grad(set_to_none=True)
    micro = 0  # micro-batches accumulated toward the current optimizer step
    done = False

    for epoch in range(args.epochs):
        if done:
            break
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=args.amp):
                loss, _, _ = model(imgs)
            # Scale so accumulated grads average (not sum) over the micro-batches.
            scaler.scale(loss / accum).backward()
            micro += 1

            if micro < accum:
                continue  # keep accumulating; no optimizer step yet
            micro = 0

            cur_lr = lr_at(step, total_steps, base_lr, warmup)
            for g in optim.param_groups:
                g["lr"] = cur_lr
            scaler.step(optim)
            if muon is not None:
                cur_muon_lr = lr_at(step, total_steps, muon_base_lr, warmup)
                for g in muon.param_groups:
                    g["lr"] = cur_muon_lr
                # GradScaler supports stepping several optimizers per update():
                # this call auto-unscales Muon's grads and inf/nan-checks them
                # independently of optim, exactly like the AdamW step above.
                scaler.step(muon)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if muon is not None:
                muon.zero_grad(set_to_none=True)

            if step % 20 == 0:
                rate = (step + 1) / (time.time() - t0)
                cmmd_str = ""
                if model.last_cmmd_loss is not None:
                    cmmd_str = f" | cmmd {model.last_cmmd_loss.item():.4f}"
                log(f"epoch {epoch:3d} | step {step:5d}/{total_steps} "
                      f"| loss {loss.item():.4f}{cmmd_str} | lr {cur_lr:.2e} "
                      f"| {rate:.1f} opt-it/s")

            step += 1
            if val_loader is not None and step % args.eval_every == 0:
                m = evaluate(model, val_loader, device, args.eval_batches,
                             args.amp, normalized=normalize)
                if m:
                    is_best = m["psnr_norm"] > best_psnr
                    log(f"  VAL step {step:5d} | loss {m['loss']:.4f} "
                        f"| psnr_norm {m['psnr_norm']:.2f}dB "
                        f"| psnr_pix {m['psnr_pix']:.2f}dB "
                        f"| ssim {m['ssim']:.4f}{'  <-- best' if is_best else ''}")
                    # Keep the best-scoring weights, not just the last ones: a
                    # short run may well peak before its cosine schedule ends,
                    # and Phase 2 should fork the best encoder available.
                    if is_best:
                        best_psnr = m["psnr_norm"]
                        tmp = f"{best_path}.tmp"
                        torch.save(
                            {
                                "backbone_state_dict": model.encoder.state_dict(),
                                "model_state_dict": model.state_dict(),
                                "mae_config": vars(model.config),
                                "config": vars(args),
                                "step": step,
                                "val_metrics": m,
                            },
                            tmp,
                        )
                        Path(tmp).replace(best_path)
            if args.ckpt_every and step % args.ckpt_every == 0:
                save_checkpoint(ckpt_path, model, optim, muon, scaler, step, args)
                log(f"  checkpoint -> {ckpt_path} (step {step})")
            if step >= total_steps:
                done = True
                break

    if val_loader is not None:
        m = evaluate(model, val_loader, device, 0, args.amp, normalized=normalize)
        if m:
            log(f"FINAL VAL (full split) | loss {m['loss']:.4f} "
                  f"| psnr_norm {m['psnr_norm']:.2f}dB "
                  f"| psnr_pix {m['psnr_pix']:.2f}dB | ssim {m['ssim']:.4f}")

    backbone = model.export_backbone()
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "model_state_dict": model.state_dict(),
            # mae_config is what visualize_mae.py needs to rebuild the exact
            # architecture; `config` is the CLI args, kept for provenance.
            "mae_config": vars(model.config),
            "config": vars(args),
        },
        args.out,
    )
    log(f"saved backbone + full model -> {Path(args.out).resolve()}")
    if best_psnr > float("-inf"):
        log(f"best held-out psnr_norm {best_psnr:.2f}dB -> {Path(best_path).resolve()}")
        log("(use the .best checkpoint for Phase 2 unless the final one scored higher)")


if __name__ == "__main__":
    main()