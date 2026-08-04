"""
Training script for the DIST student model (SAR-free cloud removal, relaxed
distillation match).

Fork of train_student.py — the only differences are the extra DIST flags below.
Reuses StudentTrainOptions/parse_hook_layers from train_student.py so the shared
flags stay in one place.

Extra flags over train_student.py:
  --dist_beta    weight on L_inter (inter-channel relation, paper Eq. 8)
  --dist_gamma   weight on L_intra (intra-channel relation, paper Eq. 9)
  --dist_act     {none,sigmoid,softmax} squash applied before correlating
  --dist_intra_min_positions  skip L_intra at layers with fewer spatial
                              locations than this (Pearson over ~4 samples is noise)
  --dist_eps     denominator epsilon in the correlation

Note --gamma_hall is inherited and now means "weight on the *exact-match*
Hoffman term".  Set it to 0 to train with DIST alone; leave it nonzero to blend.

Usage (run from khushi/student/):
    python train_student_dist.py \\
        --dataroot         ../../SEN12MSCR_student \\
        --name             sen12mscr_student_dist_v1 \\
        --model            pix2pix_attn_student_dist \\
        --dataset_mode     unaligned_sar \\
        --checkpoints_dir  ./outputs/checkpoints \\
        --teacher_checkpoints_dir ./outputs/checkpoints \\
        --teacher_name     sen12mscr_teacher_v1 \\
        --teacher_which_epoch latest \\
        --gamma_hall 0 --dist_beta 2 --dist_gamma 2 \\
        --niter 50 --niter_decay 50 \\
        --gpu_ids 0 --batchSize 1 --no_dropout --no_flip \\
        --display_id 0 --no_html --save_epoch_freq 10
"""

import time
import random

from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from train_student import StudentTrainOptions, parse_hook_layers

random.seed(10)


class StudentDISTTrainOptions(StudentTrainOptions):
    """StudentTrainOptions plus the DIST relational-loss hyperparameters."""

    def initialize(self):
        StudentTrainOptions.initialize(self)

        self.parser.add_argument(
            '--dist_beta', type=float, default=2.0,
            help='Weight on L_inter (inter-channel relation, DIST Eq. 8). '
                 'Paper uses beta=2 for classification; here it competes with an '
                 'L_G whose L1 term carries weight 100, so expect to raise it. '
                 'Tune the same way gamma_hall was: watch the first ~100 iters '
                 'and scale so beta*L_inter is comparable to 10*L_G_GAN.')
        self.parser.add_argument(
            '--dist_gamma', type=float, default=2.0,
            help='Weight on L_intra (intra-channel relation, DIST Eq. 9). '
                 'See --dist_beta for scaling guidance.')
        self.parser.add_argument(
            '--dist_act', type=str, default='none',
            choices=['none', 'sigmoid', 'softmax'],
            help="Squash applied to activations before correlating. 'none' is "
                 "the principled default (Pearson already absorbs scale/shift). "
                 "'sigmoid' matches the baseline's input exactly, isolating the "
                 "metric change. 'softmax' is the literal DIST analogue.")
        self.parser.add_argument(
            '--dist_intra_min_positions', type=int, default=16,
            help='Skip L_intra at hooked layers with fewer than this many spatial '
                 'positions (B*H*W). At batchSize 1 the deepest unet_256 block is '
                 '2x2 = 4 positions, where a Pearson coefficient is meaningless.')
        self.parser.add_argument(
            '--dist_eps', type=float, default=1e-8,
            help='Epsilon added to the correlation denominator. Guards against '
                 'dead/constant channels, which the teacher is known to have.')


def main():
    opt = StudentDISTTrainOptions().parse()
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
