"""
Quantitative val-set comparison between two trained student checkpoints
(e.g. the exact-match baseline `pix2pix_attn_student` vs the relaxed-match
`pix2pix_attn_student_dist`).

Computes PSNR / SSIM / L1 of fake_B against real_B (cloud-free ground truth) on
the val split of SEN12MSCR_student, run through the SAME data loader / same
`test()` codepath used at inference (SAR-free — only real_A is consumed).

This exists because test_student.py only dumps images to results/<name>/web/;
it never scores them. `metrics.py` in this dir is upstream's throwaway script
with hardcoded results/gt4 paths — unrelated, don't use it for this.

Usage (run from khushi/student/):
    python eval_compare.py \\
        --dataroot   ../../SEN12MSCR_student \\
        --checkpoints_dir ./outputs/checkpoints \\
        --names      sen12mscr_student_v1 sen12mscr_student_dist_v1 \\
        --model      pix2pix_attn_student pix2pix_attn_student_dist \\
        --which_epoch latest
"""

import argparse
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model


class EvalOptions(TestOptions):
    """TestOptions plus the student-model fields, plus this script's own
    multi-checkpoint arguments. Mirrors StudentTestOptions from test_student.py."""

    def initialize(self):
        TestOptions.initialize(self)
        self.parser.add_argument('--teacher_checkpoints_dir', type=str, default='./outputs/checkpoints')
        self.parser.add_argument('--teacher_name', type=str, default='sen12mscr_teacher_v1')
        self.parser.add_argument('--teacher_which_epoch', type=str, default='latest')
        self.parser.add_argument('--gamma_hall', type=float, default=100.0)
        self.parser.add_argument('--lambda3', type=float, default=1.0)
        self.parser.add_argument('--hook_layers', type=str, default='3,4,5,6,7')
        self.parser.add_argument('--dist_beta', type=float, default=2.0)
        self.parser.add_argument('--dist_gamma', type=float, default=2.0)
        self.parser.add_argument('--dist_act', type=str, default='none')
        self.parser.add_argument('--dist_intra_min_positions', type=int, default=16)
        self.parser.add_argument('--dist_eps', type=float, default=1e-8)
        # This script's own flags — parsed out before TestOptions.parse() sees argv.
        self.parser.add_argument('--names', type=str, nargs='+', required=True,
                                  help='One --name per checkpoint to evaluate.')
        self.parser.add_argument('--models', type=str, nargs='+', required=True,
                                  help='One --model key per checkpoint (same order as --names).')


def to_uint8_hwc(tensor):
    """(1,3,H,W) in [-1,1] -> (H,W,3) uint8 in [0,255]. Same mapping as util.tensor2im."""
    x = tensor[0].detach().cpu().float().numpy()
    x = (np.transpose(x, (1, 2, 0)) + 1.0) / 2.0 * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def evaluate(opt, name, model_key):
    opt.name  = name
    opt.model = model_key
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    l1s, psnrs, ssims = [], [], []
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            model.set_input(data)
            model.test()  # SAR-free codepath — only real_A is consumed

            fake_B = model.fake_B[0].detach().cpu().float().numpy()
            real_B = model.real_B[0].detach().cpu().float().numpy()
            l1s.append(np.mean(np.abs(fake_B - real_B)))  # in [-1,1] units

            fake_im = to_uint8_hwc(model.fake_B)
            real_im = to_uint8_hwc(model.real_B)
            psnrs.append(psnr(real_im, fake_im, data_range=255))
            ssims.append(ssim(real_im, fake_im, channel_axis=2, data_range=255))

    n = len(l1s)
    return {
        'n': n,
        'l1_mean':   float(np.mean(l1s)),
        'psnr_mean': float(np.mean(psnrs)),
        'psnr_std':  float(np.std(psnrs)),
        'ssim_mean': float(np.mean(ssims)),
        'ssim_std':  float(np.std(ssims)),
    }


def main():
    opt = EvalOptions().parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    names, models = opt.names, opt.models
    if len(names) != len(models):
        raise ValueError('--names and --models must have the same length, one model key per name '
                          '(got %d names, %d models)' % (len(names), len(models)))

    results = {}
    for name, model_key in zip(names, models):
        print('\n=== Evaluating %s (model=%s) on phase=%s ===' % (name, model_key, opt.phase))
        results[name] = evaluate(opt, name, model_key)

    print('\n%-32s %6s %10s %10s %10s' % ('name', 'n', 'L1', 'PSNR(dB)', 'SSIM'))
    for name, r in results.items():
        print('%-32s %6d %10.4f %6.2f±%.2f %6.4f±%.4f' %
              (name, r['n'], r['l1_mean'], r['psnr_mean'], r['psnr_std'],
               r['ssim_mean'], r['ssim_std']))

    if len(names) == 2:
        a, b = names
        print('\nDelta (%s - %s): L1 %+.4f  PSNR %+.2f dB  SSIM %+.4f' % (
            b, a,
            results[b]['l1_mean']   - results[a]['l1_mean'],
            results[b]['psnr_mean'] - results[a]['psnr_mean'],
            results[b]['ssim_mean'] - results[a]['ssim_mean']))


if __name__ == '__main__':
    main()
