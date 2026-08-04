"""
Training script for the student hallucination model (SAR-free cloud removal).

Mirrors train.py but adds student-specific CLI flags:
  --teacher-checkpoints-dir  root of teacher checkpoint tree
  --teacher-name             teacher experiment name (subfolder under checkpoints dir)
  --teacher-which-epoch      epoch label to load ('latest' or int)
  --gamma-hall               weight on the hallucination loss (default 100)
  --lambda3                  weight on SSIM in the hallucination module's translation
                             loss, i.e. L_H = L1 + lambda3 * SSIM  (default 1.0)
  --hook-layers              comma-separated UNet block indices to apply hallucination
                             loss (0=outermost); default '3,4,5,6,7'

Usage (run from AttentionGAN-for-Cloud-removal/):
    python train_student.py \\
        --dataroot         ../SEN12MSCR_student \\
        --name             sen12mscr_student_v1 \\
        --model            pix2pix_attn_student \\
        --dataset_mode     unaligned_sar \\
        --checkpoints_dir  ./outputs/checkpoints \\
        --teacher_checkpoints_dir ./outputs/checkpoints \\
        --teacher_name     sen12mscr_teacher_v1 \\
        --teacher_which_epoch latest \\
        --niter 50 --niter_decay 50 \\
        --gpu_ids 0 --batchSize 1 --no_dropout --no_flip \\
        --display_id 0 --no_html --save_epoch_freq 10
"""

import time
import argparse
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import random

random.seed(10)


class StudentTrainOptions(TrainOptions):
    """Extends TrainOptions with student-specific hyperparameters."""

    def initialize(self):
        TrainOptions.initialize(self)

        # ── Teacher checkpoint source ─────────────────────────────────────────
        self.parser.add_argument(
            '--teacher_checkpoints_dir', type=str,
            default='./outputs/checkpoints',
            help='Root directory containing teacher checkpoint subfolders')
        self.parser.add_argument(
            '--teacher_name', type=str,
            default='sen12mscr_teacher_v1',
            help='Teacher experiment name (subfolder under teacher_checkpoints_dir)')
        self.parser.add_argument(
            '--teacher_which_epoch', type=str, default='latest',
            help="Teacher checkpoint epoch to load ('latest' or epoch number)")

        # ── Hallucination loss ────────────────────────────────────────────────
        self.parser.add_argument(
            '--gamma_hall', type=float, default=100.0,
            help='Weight on the total hallucination loss.  '
                 'Default 100; tune so L_hall contribution ~10x an individual '
                 'standard loss term in the first few iterations.')
        self.parser.add_argument(
            '--lambda3', type=float, default=1.0,
            help='SSIM weight in hallucination module translation loss '
                 '(L_H = L1 + lambda3 * SSIM)')
        self.parser.add_argument(
            '--hook_layers', type=str, default='3,4,5,6,7',
            help='Comma-separated UNet block indices at which to apply the '
                 'hallucination loss.  0 = outermost block.  For unet_256 '
                 '(8 levels), valid range is 0-7.')


def parse_hook_layers(opt):
    """Convert opt.hook_layers from '3,4,5,6,7' to [3, 4, 5, 6, 7]."""
    if isinstance(opt.hook_layers, str):
        opt.hook_layers = [int(x.strip()) for x in opt.hook_layers.split(',') if x.strip()]
    return opt


def main():
    opt = StudentTrainOptions().parse()
    opt = parse_hook_layers(opt)

    data_loader  = CreateDataLoader(opt)
    dataset      = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model      = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

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

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


if __name__ == '__main__':
    main()
