"""
Experimental proof of whether the teacher is a good or bad SAR encoder.

Four experiments run on the full val split:

1. CORRELATION fake_C ↔ real_C
   If the teacher's SAR→optical translation (netG2) actually uses SAR,
   fake_C should look like real_C. Pearson r per tile, then averaged.

2. CORRELATION fake_C ↔ real_B
   fake_C is supervised with L1(fake_C, real_B). So it tries to look like
   the clear optical image. This measures how well that supervision worked.

3. SAR ABLATION — how much does SAR change the output?
   Run teacher twice per tile: once with real_C, once with zeros for SAR.
   Measure ||fake_B_real - fake_B_zero|| / ||fake_B_real|| (relative change).
   If near zero → teacher output is almost independent of SAR.

4. ATTENTION ANALYSIS
   att_A is the mask that blends netG's output with real_A.
   High att_A in cloud regions → teacher is fixing clouds.
   Low att_A everywhere → the blending is doing nothing.

5. TEACHER vs FLAT BASELINE PSNR
   Compare teacher's fake_B PSNR against simply returning real_A (the cloudy
   input) and against returning the per-image mean colour.
   If teacher barely beats a trivial baseline, it isn't doing real cloud removal.

Usage (from khushi/student/ with venv active):
    python teacher_sar_diagnostic.py [--n-tiles N]
"""

import argparse, warnings
import numpy as np
import torch
from torch.autograd import Variable
warnings.filterwarnings('ignore')

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
        gamma_hall=100.0, lambda3=1.0, hook_layers='3,4,5,6,7',
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


def pearson_r(a, b):
    """Pearson correlation between two flattened arrays."""
    a, b = a.flatten().astype(np.float32), b.flatten().astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def psnr(pred, gt):
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32))**2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0**2 / mse)


def t2np(tensor):
    """(1,C,H,W) in [-1,1] → (C,H,W) float32 in [0,1]."""
    x = tensor[0].detach().cpu().float().numpy()
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def teacher_forward(model, real_C_tensor):
    """Run teacher with a given SAR tensor (allows ablation)."""
    real_A = Variable(model.input_A)
    real_C = Variable(real_C_tensor)
    att_A  = model.netA.forward(real_A)
    fake_C = model.netG2.forward(real_C)
    fake_B_raw = model.netG.forward(torch.cat([real_A, fake_C], dim=1))
    fake_B = model.mask_layer(fake_B_raw, real_A, att_A)
    return fake_B, fake_C, att_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tiles', type=int, default=635)
    args = ap.parse_args()

    from data.data_loader import CreateDataLoader
    from models.models import create_model

    print('Loading dataset …')
    opt = make_opt()
    dataset = list(CreateDataLoader(opt).load_data())
    n = min(args.n_tiles, len(dataset))
    print(f'  Using {n} val tiles\n')

    print('Loading teacher …')
    teacher = create_model(opt)

    # accumulators
    corr_fakeC_realC = []   # exp 1: does fake_C resemble SAR?
    corr_fakeC_realB = []   # exp 2: does fake_C resemble clear optical?
    sar_ablation_rel  = []  # exp 3: relative output change when SAR=0
    att_means         = []  # exp 4: attention map mean
    psnr_teacher      = []  # exp 5a: teacher PSNR
    psnr_cloudy       = []  # exp 5b: trivial baseline — return cloudy input
    psnr_mean_colour  = []  # exp 5c: trivial baseline — per-image mean colour

    print(f'Running {n} tiles …')
    with torch.no_grad():
        for i in range(n):
            data = dataset[i]
            teacher.set_input(data)

            # --- normal forward with real SAR ---
            fake_B, fake_C, att_A = teacher_forward(teacher, teacher.input_C)
            real_A_np = t2np(teacher.input_A)   # cloudy optical
            real_B_np = t2np(teacher.input_B)   # clear optical GT
            real_C_np = t2np(teacher.input_C)   # SAR input
            fake_C_np = t2np(fake_C)
            fake_B_np = t2np(fake_B)
            att_np    = att_A[0].detach().cpu().float().numpy()  # (1,H,W)

            # exp 1: correlation fake_C ↔ real_C (per channel, then mean)
            r_fc_rc = np.mean([pearson_r(fake_C_np[c], real_C_np[c])
                               for c in range(3)])
            corr_fakeC_realC.append(r_fc_rc)

            # exp 2: correlation fake_C ↔ real_B
            r_fc_rb = np.mean([pearson_r(fake_C_np[c], real_B_np[c])
                               for c in range(3)])
            corr_fakeC_realB.append(r_fc_rb)

            # exp 3: SAR ablation — run with zeros
            zero_C = torch.zeros_like(teacher.input_C)
            fake_B_zero, _, _ = teacher_forward(teacher, zero_C)
            fake_B_zero_np = t2np(fake_B_zero)
            diff = np.abs(fake_B_np - fake_B_zero_np).mean()
            norm = np.abs(fake_B_np).mean() + 1e-8
            sar_ablation_rel.append(diff / norm)

            # exp 4: attention mean
            att_means.append(float(att_np.mean()))

            # exp 5: PSNR comparisons (uint8)
            gt_u8      = (real_B_np * 255).astype(np.uint8)
            teacher_u8 = (fake_B_np * 255).astype(np.uint8)
            cloudy_u8  = (real_A_np * 255).astype(np.uint8)
            mean_col   = real_A_np.mean(axis=(1, 2), keepdims=True)
            mean_u8    = np.broadcast_to((mean_col * 255).astype(np.uint8),
                                          real_A_np.shape)

            psnr_teacher.append(psnr(teacher_u8, gt_u8))
            psnr_cloudy.append(psnr(cloudy_u8,  gt_u8))
            psnr_mean_colour.append(psnr(mean_u8, gt_u8))

            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{n}')

    # ── results ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('TEACHER SAR DIAGNOSTIC RESULTS')
    print('='*60)

    print('\n--- Experiment 1: fake_C ↔ real_C correlation ---')
    print('(If netG2 actually encodes SAR, fake_C should resemble real_C)')
    fc_rc = np.array(corr_fakeC_realC)
    print(f'  Mean Pearson r : {fc_rc.mean():+.4f}')
    print(f'  Std            : {fc_rc.std():.4f}')
    print(f'  Range          : [{fc_rc.min():+.4f}, {fc_rc.max():+.4f}]')
    print(f'  Tiles with r>0 : {(fc_rc>0).sum()}/{n}  '
          f'({100*(fc_rc>0).mean():.1f}%)')
    if fc_rc.mean() < 0.1:
        print('  VERDICT: near-zero / negative — netG2 output is UNCORRELATED '
              'with SAR input')

    print('\n--- Experiment 2: fake_C ↔ real_B correlation ---')
    print('(netG2 is trained with L1(fake_C, real_B) — does it produce '
          'optical-like output?)')
    fc_rb = np.array(corr_fakeC_realB)
    print(f'  Mean Pearson r : {fc_rb.mean():+.4f}')
    print(f'  Std            : {fc_rb.std():.4f}')
    print(f'  Range          : [{fc_rb.min():+.4f}, {fc_rb.max():+.4f}]')

    print('\n--- Experiment 3: SAR ablation ---')
    print('(How much does zeroing the SAR input change the final output?)')
    abl = np.array(sar_ablation_rel)
    print(f'  Mean relative change : {abl.mean():.4f}  '
          f'({abl.mean()*100:.2f}% of output magnitude)')
    print(f'  Std                  : {abl.std():.4f}')
    print(f'  Max change           : {abl.max():.4f}')
    if abl.mean() < 0.05:
        print('  VERDICT: <5% change — output is nearly INDEPENDENT of SAR')
    elif abl.mean() < 0.15:
        print('  VERDICT: small but nonzero — SAR has weak influence')
    else:
        print('  VERDICT: SAR has meaningful influence on output')

    print('\n--- Experiment 4: attention map mean ---')
    print('(att_A blends netG output with cloudy input; '
          'higher = more correction)')
    att = np.array(att_means)
    print(f'  Mean attention : {att.mean():.4f}')
    print(f'  Std            : {att.std():.4f}')
    print(f'  Range          : [{att.min():.4f}, {att.max():.4f}]')
    if att.mean() < 0.1:
        print('  VERDICT: attention is near-zero — teacher mostly returns '
              'the cloudy input unchanged')

    print('\n--- Experiment 5: PSNR comparison ---')
    print('(Does the teacher beat trivial baselines?)')
    pt  = np.array(psnr_teacher)
    pc  = np.array(psnr_cloudy)
    pm  = np.array(psnr_mean_colour)
    print(f'  Teacher output     : {pt.mean():.2f} ± {pt.std():.2f} dB')
    print(f'  Cloudy input as-is : {pc.mean():.2f} ± {pc.std():.2f} dB  '
          f'(baseline: do nothing)')
    print(f'  Per-image mean col : {pm.mean():.2f} ± {pm.std():.2f} dB  '
          f'(baseline: flat colour)')
    print(f'  Teacher gain over "do nothing" : '
          f'{pt.mean()-pc.mean():+.2f} dB')
    print(f'  Teacher gain over mean colour  : '
          f'{pt.mean()-pm.mean():+.2f} dB')

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    verdicts = []
    if fc_rc.mean() < 0.1:
        verdicts.append('fake_C uncorrelated with SAR (Exp 1)')
    if abl.mean() < 0.10:
        verdicts.append('zeroing SAR barely changes output (Exp 3)')
    if att.mean() < 0.15:
        verdicts.append('attention nearly always near-zero (Exp 4)')
    if pt.mean() - pc.mean() < 1.0:
        verdicts.append('teacher barely beats "do nothing" baseline (Exp 5)')
    if verdicts:
        print('\n  Evidence teacher is a poor SAR encoder:')
        for v in verdicts:
            print(f'    ✗  {v}')
    else:
        print('\n  Teacher appears to use SAR meaningfully.')


if __name__ == '__main__':
    main()
