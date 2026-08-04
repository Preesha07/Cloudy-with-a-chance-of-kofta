"""
Visual comparison: Baseline (ep20) vs FSP v1 (ep10) on val tiles.

Saves a grid PNG to viz_out/fsp_comparison.png:
  rows = sample tiles
  cols = cloudy | baseline | FSP v1 | ground truth

Usage (from khushi/student/):
    python visualize_fsp.py [--num-samples N] [--out viz_out/fsp_comparison.png]
"""
import argparse, os, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

DATAROOT = '/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student'
CKPT_DIR = './outputs/checkpoints'


def make_opt(name, model_key, epoch):
    import argparse as _ap
    opt = _ap.Namespace(
        dataroot=DATAROOT, name=name, model=model_key,
        checkpoints_dir=CKPT_DIR, which_epoch=str(epoch),
        teacher_checkpoints_dir=CKPT_DIR,
        teacher_name='sen12mscr_teacher_v1', teacher_which_epoch='latest',
        gamma_hall=100.0, lambda3=1.0, hook_layers='3,4,5,6,7',
        dist_beta=2.0, dist_gamma=2.0, dist_act='none',
        dist_intra_min_positions=16, dist_eps=1e-8,
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
    return opt


def t2rgb(tensor):
    """(1,3,H,W) in [-1,1] → (H,W,3) float32 in [0,1] for matplotlib."""
    x = tensor[0].detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0 ** 2) / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=6)
    parser.add_argument('--out', type=str, default='viz_out/fsp_comparison.png')
    parser.add_argument('--baseline-epoch', type=int, default=20)
    parser.add_argument('--fsp-epoch',      type=int, default=10)
    parser.add_argument('--baseline-name',  type=str, default='sen12mscr_student_v1')
    parser.add_argument('--fsp-name',       type=str, default='sen12mscr_student_fsp_v1')
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    from data.data_loader import CreateDataLoader
    from models.models import create_model

    print('Loading dataset …')
    opt_b = make_opt(args.baseline_name, 'pix2pix_attn_student', args.baseline_epoch)
    dataset = list(CreateDataLoader(opt_b).load_data())

    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(dataset), size=args.num_samples, replace=False).tolist())
    print(f'Selected tile indices: {indices}')

    print(f'Loading baseline ({args.baseline_name} ep{args.baseline_epoch}) …')
    model_b = create_model(opt_b)

    print(f'Loading FSP model ({args.fsp_name} ep{args.fsp_epoch}) …')
    opt_f = make_opt(args.fsp_name, 'pix2pix_attn_student_fsp', args.fsp_epoch)
    model_f = create_model(opt_f)

    # --- layout ---
    n = len(indices)
    cols = 4  # cloudy | baseline | FSP | GT
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.8, n * 2.9),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.28})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles  = ['Cloudy input', f'Baseline  ep {args.baseline_epoch}',
                   f'FSP v1  ep {args.fsp_epoch}  ★', 'Ground truth']
    col_colours = ['#8b949e', '#58a6ff', '#f0883e', '#3fb950']

    for col, (title, colour) in enumerate(zip(col_titles, col_colours)):
        axes[0, col].set_title(title, fontsize=7.5, color=colour,
                               fontweight='bold', pad=5, loc='center')

    print('Running inference …')
    with torch.no_grad():
        for row, idx in enumerate(indices):
            data = dataset[idx]

            model_b.set_input(data)
            model_b.test()
            cloudy_rgb = t2rgb(model_b.real_A)
            gt_rgb     = t2rgb(model_b.real_B)
            base_rgb   = t2rgb(model_b.fake_B)

            model_f.set_input(data)
            model_f.test()
            fsp_rgb    = t2rgb(model_f.fake_B)

            # PSNR relative to GT (uint8 scale)
            gt_u8   = (gt_rgb   * 255).astype(np.uint8)
            base_u8 = (base_rgb * 255).astype(np.uint8)
            fsp_u8  = (fsp_rgb  * 255).astype(np.uint8)
            base_p  = psnr(gt_u8, base_u8)
            fsp_p   = psnr(gt_u8, fsp_u8)

            images = [cloudy_rgb, base_rgb, fsp_rgb, gt_rgb]
            for col, img in enumerate(images):
                ax = axes[row, col]
                ax.imshow(img, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # tile index label on left edge
                if col == 0:
                    ax.set_ylabel(f'#{idx:03d}', fontsize=6.5, color='#7d8590',
                                  rotation=0, labelpad=28, va='center')

                # PSNR annotation below baseline and FSP cells
                if col == 1:
                    colour = '#3fb950' if base_p >= fsp_p else '#f85149'
                    ax.set_xlabel(f'{base_p:.2f} dB', fontsize=6.5,
                                  color=colour, labelpad=3)
                elif col == 2:
                    colour = '#3fb950' if fsp_p >= base_p else '#f85149'
                    ax.set_xlabel(f'{fsp_p:.2f} dB', fontsize=6.5,
                                  color=colour, labelpad=3)
                else:
                    ax.set_xlabel('', labelpad=3)

            # amber border on FSP column
            axes[row, 2].spines['bottom'].set_visible(True)
            axes[row, 2].spines['bottom'].set_color('#f0883e')
            axes[row, 2].spines['bottom'].set_linewidth(1.5)
            axes[row, 2].spines['top'].set_visible(True)
            axes[row, 2].spines['top'].set_color('#f0883e')
            axes[row, 2].spines['top'].set_linewidth(1.5)

            print(f'  tile {idx:3d}  baseline {base_p:.2f} dB  FSP {fsp_p:.2f} dB'
                  f'  {"FSP wins" if fsp_p > base_p else "base wins"}')

    fig.patch.set_facecolor('#0d1117')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('#161b22')

    fig.text(0.5, 0.01,
             'False-colour R/G/NIR · no SAR used at inference · green = higher PSNR on this tile',
             ha='center', fontsize=6.5, color='#7d8590')

    plt.savefig(args.out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
