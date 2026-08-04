"""
SAR-guided teacher vs optical-only student visual comparison.

Columns: cloudy input | teacher (SAR access) | student (no SAR) | ground truth

Each row is a val tile.  Tiles with heavy cloud cover are selected by looking
for low mean intensity in real_A (proxy for thick cloud fraction).

Usage (from khushi/student/ with venv active):
    python visualize_sar_vs_nosar.py [--num-samples N] [--out viz_out/sar_vs_nosar.png]
    python visualize_sar_vs_nosar.py --tiles 5 100 250 400   # exact tile indices

Teacher checkpoint:  outputs/checkpoints/sen12mscr_teacher_v1/latest_net_*.pth
Student checkpoint:  outputs/checkpoints/sen12mscr_student_v1/<epoch>_net_*.pth
"""
import argparse, os, sys, warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CKPT_DIR  = './outputs/checkpoints'
DATAROOT  = '/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student'


# ── option namespaces ──────────────────────────────────────────────────────────

def _base_opt(name, model_key, epoch):
    import argparse as _ap
    return _ap.Namespace(
        dataroot=DATAROOT, name=name, model=model_key,
        checkpoints_dir=CKPT_DIR, which_epoch=str(epoch),
        # teacher-side fields (needed at student init time)
        teacher_checkpoints_dir=CKPT_DIR,
        teacher_name='sen12mscr_teacher_v1', teacher_which_epoch='latest',
        gamma_hall=100.0, lambda3=1.0, hook_layers='3,4,5,6,7',
        dist_beta=2.0, dist_gamma=2.0, dist_act='none',
        dist_intra_min_positions=16, dist_eps=1e-8,
        cross_depth_weight=2.0, fsp_distance='pearson',
        fsp_pairs='adjacent', fsp_min_positions=17,
        # data
        dataset_mode='unaligned_sar', phase='val',
        batchSize=1, nThreads=0, serial_batches=True, no_flip=True,
        loadSize=256, fineSize=256, resize_or_crop='resize_and_crop',
        max_dataset_size=float('inf'),
        # network
        input_nc=3, output_nc=3, ngf=64, ndf=64,
        which_model_netG='unet_256', which_model_netG2='unet_256',
        which_model_netA='unet_256', which_model_netD='basic',
        n_layers_D=3, norm='instance', no_dropout=True,
        init_type='xavier', no_lsgan=False, pool_size=50,
        isTrain=False, gpu_ids=[0],
        which_direction='AtoB', which_direction_model='AtoB',
        continue_train=False, display_id=0, display_freq=500, display_winsize=256,
    )


def teacher_opt():
    return _base_opt('sen12mscr_teacher_v1', 'pix2pix_attn', 'latest')


def student_opt(epoch=20):
    return _base_opt('sen12mscr_student_v1', 'pix2pix_attn_student', epoch)


# ── teacher forward (bypasses deprecated volatile Variable) ────────────────────

def teacher_forward(model, data):
    """Run teacher inference with SAR.  Returns fake_B (H,W,3) uint8."""
    model.set_input(data)
    with torch.no_grad():
        real_A = model.real_A if hasattr(model, 'real_A') else None
        # trigger set_input so tensors are populated
        # then run the forward pass manually
        from torch.autograd import Variable
        real_A = Variable(model.input_A)
        real_C = Variable(model.input_C)
        att_A  = model.netA.forward(real_A)
        fake_C = model.netG2.forward(real_C)
        fake_B_raw = model.netG.forward(torch.cat([real_A, fake_C], dim=1))
        fake_B = model.mask_layer(fake_B_raw, real_A, att_A)
        model.fake_B = fake_B
        model.real_A = real_A
        model.real_B = Variable(model.input_B)
        model.real_C = real_C
    return t2u8(model.fake_B)


# ── helpers ────────────────────────────────────────────────────────────────────

def t2u8(tensor):
    """(1,3,H,W) in [-1,1] → (H,W,3) uint8."""
    x = tensor[0].detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))
    return np.clip((x + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def cloudiness(img_u8):
    """Higher mean → brighter (more cloud-white) in the scene."""
    return img_u8.mean()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-samples',    type=int,   default=6)
    ap.add_argument('--tiles',          type=int,   nargs='+', default=None,
                    help='Explicit tile indices; overrides --num-samples and --cloudy-bias')
    ap.add_argument('--student-epoch',  type=int,   default=20)
    ap.add_argument('--out',            type=str,   default='viz_out/sar_vs_nosar.png')
    ap.add_argument('--cloudy-bias',    action='store_true', default=True,
                    help='Pick tiles that are on the cloudier end of the val split')
    ap.add_argument('--scan-first',     type=int,   default=300,
                    help='How many tiles to scan when --cloudy-bias is on')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    from data.data_loader import CreateDataLoader
    from models.models import create_model

    # dataset (serial, so index i → tile i reproducibly)
    print('Loading dataset …')
    ds_opt = teacher_opt()
    dataset = list(CreateDataLoader(ds_opt).load_data())
    print(f'  {len(dataset)} val tiles available')

    # pick tiles
    if args.tiles:
        indices = args.tiles
    elif args.cloudy_bias:
        scan = min(args.scan_first, len(dataset))
        brightness = []
        for i in range(scan):
            data = dataset[i]
            a = data['A']
            if isinstance(a, (list, tuple)):
                a = a[0]
            brightness.append((cloudiness(t2u8(a)), i))
        # pick the cloudiest (brightest proxy)
        brightness.sort(reverse=True)
        step = max(1, len(brightness) // (args.num_samples * 4))
        candidates = [idx for _, idx in brightness]
        # spread across the ranking to avoid all-same-scene picks
        indices = sorted([candidates[i * step] for i in range(args.num_samples)])
        print(f'  Cloudy-biased tiles: {indices}')
    else:
        rng = np.random.default_rng(42)
        indices = sorted(rng.choice(len(dataset), size=args.num_samples, replace=False).tolist())
        print(f'  Random tiles: {indices}')

    print('Loading teacher (has SAR access) …')
    t_opt = teacher_opt()
    teacher = create_model(t_opt)

    print(f'Loading student ep{args.student_epoch} (optical only) …')
    s_opt = student_opt(args.student_epoch)
    student = create_model(s_opt)

    # ── figure ─────────────────────────────────────────────────────────────────
    n = len(indices)
    COL_W, ROW_H = 2.8, 3.0
    fig, axes = plt.subplots(n, 4, figsize=(4 * COL_W, n * ROW_H),
                              gridspec_kw={'wspace': 0.04, 'hspace': 0.32})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels  = ['Cloudy input',
                   'Teacher  (SAR ✓)',
                   f'Student  ep{args.student_epoch}  (SAR ✗)',
                   'Ground truth']
    col_colours = ['#8b949e', '#58a6ff', '#f0883e', '#3fb950']

    for c, (lbl, col) in enumerate(zip(col_labels, col_colours)):
        axes[0, c].set_title(lbl, fontsize=7.5, color=col,
                              fontweight='bold', pad=5)

    print('\nRunning inference …')
    with torch.no_grad():
        for row, idx in enumerate(indices):
            data = dataset[idx]

            # teacher (uses SAR)
            teach_u8  = teacher_forward(teacher, data)
            cloudy_u8 = t2u8(teacher.real_A)
            gt_u8     = t2u8(teacher.real_B)

            # student (optical only)
            student.set_input(data)
            student.test()
            stud_u8 = t2u8(student.fake_B)

            teach_p = psnr(teach_u8, gt_u8)
            stud_p  = psnr(stud_u8,  gt_u8)
            gap_db  = teach_p - stud_p          # positive → teacher wins

            images = [cloudy_u8, teach_u8, stud_u8, gt_u8]
            for c, img in enumerate(images):
                ax = axes[row, c]
                ax.imshow(img, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c == 0:
                    ax.set_ylabel(f'#{idx:03d}', fontsize=6.5, color='#7d8590',
                                  rotation=0, labelpad=28, va='center')
                if c == 1:
                    colour = '#3fb950' if teach_p >= stud_p else '#f85149'
                    ax.set_xlabel(f'{teach_p:.2f} dB', fontsize=6.5,
                                  color=colour, labelpad=2)
                elif c == 2:
                    colour = '#3fb950' if stud_p >= teach_p else '#f85149'
                    ax.set_xlabel(f'{stud_p:.2f} dB', fontsize=6.5,
                                  color=colour, labelpad=2)
                else:
                    ax.set_xlabel('', labelpad=2)

            # blue border on teacher column (SAR bonus is visible there)
            for side in ('top', 'bottom'):
                axes[row, 1].spines[side].set_visible(True)
                axes[row, 1].spines[side].set_color('#58a6ff')
                axes[row, 1].spines[side].set_linewidth(1.5)

            sign = '+' if gap_db >= 0 else ''
            print(f'  tile {idx:3d}  teacher {teach_p:.2f} dB  student {stud_p:.2f} dB'
                  f'  gap {sign}{gap_db:.2f} dB')

    fig.patch.set_facecolor('#0d1117')
    for row_ax in axes:
        for ax in row_ax:
            ax.set_facecolor('#161b22')

    fig.text(0.5, 0.005,
             'Teacher sees SAR (Sentinel-1) at both train AND test time · '
             'Student sees only optical at test time · green = higher PSNR',
             ha='center', fontsize=6, color='#7d8590')

    plt.savefig(args.out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
