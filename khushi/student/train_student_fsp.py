"""
Training script for the cross-depth FSP student model (SAR-free cloud removal).

Fork of train_student_dist.py — adds the cross-depth FSP flags, the three term
toggles, and per-pair epoch diagnostics.  Reuses StudentDISTTrainOptions so the
exact-match and DIST flags stay defined in one place.

Extra flags over train_student_dist.py:
  --use_hall / --no_hall               toggle the Hoffman exact-match term
  --use_dist / --no_dist               toggle the within-layer DIST terms
  --use_cross_depth / --no_cross_depth toggle the new cross-depth FSP term
  --cross_depth_weight   weight on the summed cross-depth term
  --fsp_distance         {pearson,frobenius}  distance between FSP matrices
  --fsp_pairs            {adjacent,all}       which layer pairs to build
  --fsp_min_positions    skip pairs whose deeper layer has fewer positions (0=off)
  --fsp_eps              degeneracy threshold on the flattened-G std

--gamma_hall still scales L_hall; --no_hall gates it out entirely and also skips
its compute, which --gamma_hall 0 does not.

Per-epoch cross-depth diagnostics are appended to
  <checkpoints_dir>/<name>/fsp_metrics.txt
one line per epoch, with the mean distance and degenerate fraction per layer pair.

Usage (run from khushi/student/):
    python train_student_fsp.py \\
        --dataroot         ../../SEN12MSCR_student \\
        --name             sen12mscr_student_fsp_v1 \\
        --model            pix2pix_attn_student_fsp \\
        --dataset_mode     unaligned_sar \\
        --checkpoints_dir  ./outputs/checkpoints \\
        --teacher_checkpoints_dir ./outputs/checkpoints \\
        --teacher_name     sen12mscr_teacher_v1 \\
        --teacher_which_epoch latest \\
        --gamma_hall 100 --no_dist --cross_depth_weight 10 \\
        --niter 15 --niter_decay 15 \\
        --gpu_ids 0 --batchSize 1 --no_dropout --no_flip \\
        --display_id 0 --no_html --save_epoch_freq 5
"""

import os
import time
import random

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from train_student import parse_hook_layers
from train_student_dist import StudentDISTTrainOptions

random.seed(10)


class StudentFSPTrainOptions(StudentDISTTrainOptions):
    """StudentDISTTrainOptions plus cross-depth FSP hyperparameters and toggles."""

    def initialize(self):
        StudentDISTTrainOptions.initialize(self)

        # ── Term toggles ──────────────────────────────────────────────────────
        # Paired --use_x / --no_x so any subset of the three distillation terms
        # can be ablated from the CLI without editing the model.
        self.parser.add_argument(
            '--use_hall', dest='use_hall', action='store_true',
            help='Enable the Hoffman exact-match term L_hall (default on)')
        self.parser.add_argument(
            '--no_hall', dest='use_hall', action='store_false',
            help='Disable L_hall entirely, skipping its compute '
                 '(unlike --gamma_hall 0, which still computes it)')
        self.parser.set_defaults(use_hall=True)

        self.parser.add_argument(
            '--use_dist', dest='use_dist', action='store_true',
            help='Enable the within-layer DIST relational terms (default on)')
        self.parser.add_argument(
            '--no_dist', dest='use_dist', action='store_false',
            help='Disable L_inter / L_intra')
        self.parser.set_defaults(use_dist=True)

        self.parser.add_argument(
            '--use_cross_depth', dest='use_cross_depth', action='store_true',
            help='Enable the cross-depth FSP term (default on)')
        self.parser.add_argument(
            '--no_cross_depth', dest='use_cross_depth', action='store_false',
            help='Disable the cross-depth FSP term')
        self.parser.set_defaults(use_cross_depth=True)

        # ── Cross-depth FSP hyperparameters ───────────────────────────────────
        self.parser.add_argument(
            '--cross_depth_weight', type=float, default=1.0,
            help='Weight on the summed cross-depth FSP loss. Under the pearson '
                 'distance each pair contributes 1-rho in [0,2], so the sum is '
                 'O(#pairs). Scale it against 10*L_G_GAN, the same rule used to '
                 'pick gamma_hall=100.')
        self.parser.add_argument(
            '--fsp_distance', type=str, default='pearson',
            choices=['pearson', 'frobenius'],
            help="Distance between teacher and student FSP matrices. 'pearson' "
                 "(default) is scale-invariant, which matters because C_i*C_j and "
                 "the activation magnitudes differ by orders of magnitude between "
                 "the shallow and deep pairs. 'frobenius' is the Yim et al. form.")
        self.parser.add_argument(
            '--fsp_pairs', type=str, default='adjacent',
            choices=['adjacent', 'all'],
            help="Which hooked-layer pairs to build FSP matrices between. "
                 "'adjacent' (default) gives 4 pairs for hooks 3,4,5,6,7; "
                 "'all' gives 10.")
        self.parser.add_argument(
            '--fsp_min_positions', type=int, default=0,
            help='Skip pairs whose deeper layer has fewer than this many spatial '
                 'positions. rank(G) <= H_j*W_j, and at hooks 3-7 the pairs have '
                 '256/64/16/4 positions — the two deepest build a 512x512 matrix '
                 'from just 16 and 4 samples. 0 (default) keeps all pairs; the '
                 'per-pair diagnostics show whether raising it is warranted.')
        self.parser.add_argument(
            '--fsp_eps', type=float, default=1e-6,
            help='Degeneracy threshold on the std of the flattened FSP matrix. '
                 'Instances below it are excluded from the pearson distance and '
                 'counted in the *_degen diagnostic.')

        # ── FreqKD hyperparameters (Thaker et al., arXiv 2606.11572, 2026) ───
        self.parser.add_argument(
            '--use_freqkd', dest='use_freqkd', action='store_true',
            help='Enable FreqKD frequency-decoupled distillation. Applies strict '
                 'MSE on the low-frequency band (shared scene structure) and '
                 'relaxed log-MSE on the high-frequency band (SAR speckle / '
                 'modality-specific texture netH cannot reproduce from optical).')
        self.parser.add_argument(
            '--no_freqkd', dest='use_freqkd', action='store_false',
            help='Disable FreqKD (default).')
        self.parser.set_defaults(use_freqkd=False)

        self.parser.add_argument(
            '--freqkd_weight', type=float, default=10.0,
            help='Weight on the summed FreqKD loss across active layers. '
                 'Scale so freqkd_weight * L_freqkd ~ 10 * L_G_GAN, the same '
                 'calibration rule used for cross_depth_weight.')
        self.parser.add_argument(
            '--freqkd_rc', type=float, default=0.5,
            help='Radial frequency cut-off in [0, 1]. 0.5 puts half the spectral '
                 'energy in the low-pass band. Ablated in the FreqKD paper; 0.5 '
                 'is their optimum.')
        self.parser.add_argument(
            '--freqkd_eta', type=float, default=0.1,
            help='Weight on the high-frequency band loss relative to the low band '
                 '(paper Eq. 6). 0.1 relaxes but does not zero out high-freq '
                 'supervision.')
        self.parser.add_argument(
            '--freqkd_min_spatial', type=int, default=8,
            help='Skip FreqKD at layers whose spatial size < this value. 2D FFT '
                 'on 4x4 or 2x2 grids (UNet blocks 6, 7) yields too few bins for '
                 'a meaningful low/high split. Default 8 skips those two blocks.')


def main():
    opt = StudentFSPTrainOptions().parse()
    opt = parse_hook_layers(opt)

    data_loader  = CreateDataLoader(opt)
    dataset      = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model      = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # Per-epoch FSP diagnostics land next to the checkpoints.  BaseOptions.parse()
    # has already created this directory.
    metrics_path = os.path.join(opt.checkpoints_dir, opt.name, 'fsp_metrics.txt')
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w') as fh:
            fh.write('# epoch'
                     '  fsp_I_J:<mean dist>  fsp_I_J_degen:<degen frac>'
                     '  fkd_lN_low:<mean low-band MSE>  fkd_lN_high:<mean log-MSE>\n')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        if hasattr(model, 'reset_epoch_metrics'):
            model.reset_epoch_metrics()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter  += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_steps, dataset_size)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                if hasattr(model, 'get_fsp_pair_errors'):
                    errors.update(model.get_fsp_pair_errors())
                if hasattr(model, 'get_freqkd_layer_errors'):
                    errors.update(model.get_freqkd_layer_errors())
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(
                        epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        # ── Per-epoch cross-depth diagnostics ─────────────────────────────────
        # This is the instrument for judging whether the teacher's FSP matrices
        # carry real cross-depth structure — especially at the near-rank-deficient
        # deep pairs — before the loss's gradient signal is trusted.
        if hasattr(model, 'get_epoch_metrics'):
            em = model.get_epoch_metrics()
            if em:
                line = 'epoch %d  ' % epoch + '  '.join(
                    '%s:%.6f' % (k, v) for k, v in em.items())
                print(line)
                with open(metrics_path, 'a') as fh:
                    fh.write(line + '\n')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    main()
