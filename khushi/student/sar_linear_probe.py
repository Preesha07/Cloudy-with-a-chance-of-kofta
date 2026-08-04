"""
SAR linear probe: measures how much SAR information each UNet layer encodes.

Probes the TEACHER's netG2 (the SAR encoder that FSP tries to match).

Two modes compared side-by-side:
  gap      — global average pool of block output (BROKEN by InstanceNorm:
              IN forces every channel mean → 0, so R²=0 everywhere)
  prenorm  — hooks the Conv2d inside each block BEFORE InstanceNorm is
              applied, extracts per-channel (µ, σ) statistics.
              InstanceNorm is designed to strip exactly these statistics
              (Huang & Belongie, AdaIN ICCV 2017), so probing them gives
              the true SAR encoding picture.

Usage (from khushi/student/ with venv active):
    python sar_linear_probe.py [--n-tiles N] [--out sar_probe_results.txt]
"""

import argparse, os, warnings
import numpy as np
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

CKPT_DIR = './outputs/checkpoints'
DATAROOT  = '/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student'
TEACHER   = 'sen12mscr_teacher_v1'


def make_opt():
    import argparse as _ap
    return _ap.Namespace(
        dataroot=DATAROOT, name=TEACHER, model='pix2pix_attn',
        checkpoints_dir=CKPT_DIR, which_epoch='latest',
        teacher_checkpoints_dir=CKPT_DIR,
        teacher_name=TEACHER, teacher_which_epoch='latest',
        gamma_hall=100.0, lambda3=1.0, hook_layers='0,1,2,3,4,5,6,7',
        dataset_mode='unaligned_sar', phase='val',
        batchSize=1, nThreads=0, serial_batches=True, no_flip=True,
        loadSize=256, fineSize=256, resize_or_crop='resize_and_crop',
        max_dataset_size=float('inf'),
        input_nc=3, output_nc=3, ngf=64, ndf=64,
        which_model_netG='unet_256', which_model_netG2='unet_256',
        which_model_netA='unet_256', which_model_netD='basic',
        n_layers_D=3, norm='instance', no_dropout=True,
        init_type='xavier', no_lsgan=False, pool_size=50,
        isTrain=False, gpu_ids=[0],
        which_direction='AtoB', which_direction_model='AtoB',
        continue_train=False, display_id=0, display_freq=500, display_winsize=256,
    )


def get_unet_blocks(gen):
    """Return list of UnetSkipConnectionBlock instances, outermost first."""
    from models.networks import UnetSkipConnectionBlock

    node = gen
    if isinstance(node, nn.DataParallel):
        node = node.module
    if isinstance(node, nn.Sequential):
        node = list(node.children())[0]
    if hasattr(node, 'model'):
        node = node.model

    blocks = []
    while isinstance(node, UnetSkipConnectionBlock):
        blocks.append(node)
        inner = None
        for child in node.model.children():
            if isinstance(child, UnetSkipConnectionBlock):
                inner = child
                break
        if inner is None:
            break
        node = inner
    return blocks


def find_downconv(block):
    """Return the first Conv2d in a block's model Sequential (the downconv,
    before any InstanceNorm).  This is what we hook for the pre-norm probe."""
    for child in block.model.children():
        if isinstance(child, nn.Conv2d):
            return child
    return None


def prenorm_stats(tensor):
    """Extract (µ, σ) per channel: (1,C,H,W) → (2C,) numpy vector.
    InstanceNorm zeroes µ and sets σ=1 AFTER this point.
    These statistics are exactly what IN strips — probing them gives the
    true per-sample radiometric content of each layer.
    If spatial size is 1×1, σ is undefined (single value) — use µ only.
    """
    x = tensor[0].detach().cpu().float()   # (C, H, W)
    mu = x.mean(dim=(-2, -1)).numpy()      # (C,)
    if x.shape[-1] > 1 and x.shape[-2] > 1:
        sigma = x.std(dim=(-2, -1)).numpy()
        return np.concatenate([mu, sigma])  # (2C,)
    return mu                               # (C,) — spatial 1×1, no σ


def gap_vec(tensor):
    """Global average pool: (1,C,H,W) → (C,) numpy.  Broken by IN."""
    return tensor[0].mean(dim=(-2, -1)).detach().cpu().float().numpy()


def fit_r2(features, targets, split):
    """Fit Ridge regression on features[:split] → targets[:split],
    return per-target-column R² on features[split:]."""
    r2s = []
    for c in range(targets.shape[1]):
        pipe = Pipeline([('sc', StandardScaler()), ('ridge', Ridge(alpha=1.0))])
        pipe.fit(features[:split], targets[:split, c])
        pred = pipe.predict(features[split:])
        r2s.append(max(float(r2_score(targets[split:, c], pred)), 0.0))
    return r2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tiles', type=int, default=400)
    ap.add_argument('--out',     type=str, default='sar_probe_results.txt')
    ap.add_argument('--layers',  type=str, default='0,1,2,3,4,5,6,7')
    args = ap.parse_args()

    probe_layers = [int(x) for x in args.layers.split(',')]

    from data.data_loader import CreateDataLoader
    from models.models import create_model

    print('Loading dataset …')
    opt = make_opt()
    dataset = list(CreateDataLoader(opt).load_data())
    n = min(args.n_tiles, len(dataset))
    print(f'  Using {n} of {len(dataset)} val tiles')

    print('Loading teacher model …')
    teacher = create_model(opt)

    blocks = get_unet_blocks(teacher.netG2)
    print(f'  Found {len(blocks)} UNet blocks in netG2\n')

    # ── register hooks ────────────────────────────────────────────────────────
    gap_acts    = {k: [] for k in probe_layers}   # block output (post-IN)
    prenorm_acts = {k: [] for k in probe_layers}  # conv output (pre-IN)
    hooks = []

    for k in probe_layers:
        if k >= len(blocks):
            print(f'  Warning: layer {k} out of range, skipping')
            continue

        block = blocks[k]

        # hook 1: block output (GAP, post-IN — broken baseline)
        def make_gap_hook(idx):
            def h(module, inp, out):
                gap_acts[idx].append(gap_vec(out))
            return h
        hooks.append(block.register_forward_hook(make_gap_hook(k)))

        # hook 2: downconv output (pre-IN — the fix)
        conv = find_downconv(block)
        if conv is not None:
            def make_prenorm_hook(idx):
                def h(module, inp, out):
                    prenorm_acts[idx].append(prenorm_stats(out))
                return h
            hooks.append(conv.register_forward_hook(make_prenorm_hook(k)))
        else:
            print(f'  Warning: no Conv2d found in block {k}')

    # ── run forward passes ────────────────────────────────────────────────────
    sar_targets = []
    print(f'Running {n} tiles through netG2 …')
    with torch.no_grad():
        for i in range(n):
            data = dataset[i]
            teacher.set_input(data)
            from torch.autograd import Variable
            real_C = Variable(teacher.input_C)
            teacher.netG2.forward(real_C)
            # SAR target: per-channel global mean of raw input (not IN-affected)
            sar_targets.append(gap_vec(teacher.input_C))
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{n}')

    for h in hooks:
        h.remove()

    sar_arr = np.stack(sar_targets)   # (N, 3)  VV / VH / VV-VH
    split   = int(0.8 * n)

    # ── print results table ───────────────────────────────────────────────────
    header = (f'{"Lyr":>3}  {"Spatial":>9}  '
              f'{"GAP R²(VV)":>10} {"GAP R²(VH)":>10} {"GAP mean":>8}  '
              f'{"PreNorm R²(VV)":>14} {"PreNorm R²(VH)":>14} {"PreNorm mean":>12}')
    print('\n' + header)
    print('-' * len(header))

    results_gap     = {}
    results_prenorm = {}

    for k in sorted(probe_layers):
        spatial = 256 // (2 ** (k + 1))

        # GAP probe
        if gap_acts[k]:
            arr = np.stack(gap_acts[k])
            r2g = fit_r2(arr, sar_arr, split)
            results_gap[k] = r2g
        else:
            r2g = [0.0, 0.0, 0.0]

        # Pre-norm probe
        if prenorm_acts[k]:
            arr = np.stack(prenorm_acts[k])
            r2p = fit_r2(arr, sar_arr, split)
            results_prenorm[k] = r2p
        else:
            r2p = [0.0, 0.0, 0.0]

        print(f'{k:>3}  {spatial:>5}×{spatial:<3}  '
              f'{r2g[0]:>10.4f} {r2g[1]:>10.4f} {np.mean(r2g):>8.4f}  '
              f'{r2p[0]:>14.4f} {r2p[1]:>14.4f} {np.mean(r2p):>12.4f}')

    # ── interpretation ────────────────────────────────────────────────────────
    print('\n--- Interpretation ---')
    print('GAP probe is expected to give R²≈0 everywhere due to InstanceNorm.')
    print('PreNorm probe bypasses IN — R² here reflects true SAR encoding.\n')

    prenorm_means = {k: np.mean(v) for k, v in results_prenorm.items()}
    if prenorm_means:
        best  = max(prenorm_means, key=prenorm_means.get)
        worst = min(prenorm_means, key=prenorm_means.get)
        print(f'Most SAR info (pre-norm):  layer {best}  (R²={prenorm_means[best]:.4f})')
        print(f'Least SAR info (pre-norm): layer {worst}  (R²={prenorm_means[worst]:.4f})')

        threshold = 0.05
        useful = [k for k, v in sorted(prenorm_means.items()) if v >= threshold]
        print(f'\nLayers with pre-norm R² >= {threshold}: {useful}')
        if len(useful) >= 2:
            pairs = [(useful[i], useful[i+1]) for i in range(len(useful)-1)]
            print(f'Recommended FSP pairs (evidence-based): {pairs}')

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(args.out, 'w') as f:
        f.write('layer,spatial,gap_r2_VV,gap_r2_VH,gap_r2_VVVH,gap_mean,'
                'prenorm_r2_VV,prenorm_r2_VH,prenorm_r2_VVVH,prenorm_mean\n')
        for k in sorted(probe_layers):
            spatial = 256 // (2 ** (k + 1))
            rg = results_gap.get(k, [0,0,0])
            rp = results_prenorm.get(k, [0,0,0])
            f.write(f'{k},{spatial},'
                    f'{rg[0]:.6f},{rg[1]:.6f},{rg[2]:.6f},{np.mean(rg):.6f},'
                    f'{rp[0]:.6f},{rp[1]:.6f},{rp[2]:.6f},{np.mean(rp):.6f}\n')
    print(f'\nResults saved to {args.out}')


if __name__ == '__main__':
    main()
