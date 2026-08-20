"""Training driver for the NIR teacher (pix2pix_attn_nir) with periodic
visual saves.  Identical to train.py except that every VIS_FREQ optimizer
steps it writes a grid of real_A / fake_B / real_B / att_A-heatmap /
att_A-overlay / real_C / fake_C to outputs/visuals_nir/<name>/.

Usage (from AttentionGAN-for-Cloud-removal/):
  python train_nir_vis.py <same flags as train.py>

The visualization interval is hardcoded to 4000 steps (batchSize-agnostic,
since total_steps counts images processed, not gradient steps; 4000 images
≈ 1000 optimizer steps at batchSize=4).
"""

import os
import time

import numpy as np
import torch
from PIL import Image

import random
random.seed(10)

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util

torch.backends.cudnn.benchmark = True

VIS_FREQ = 4000  # save visuals every this many images processed


def save_visuals(visuals, out_dir, prefix):
    """Save each visual in `visuals` as a PNG under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for label, img_np in visuals.items():
        if img_np is None:
            continue
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
            img_np = np.stack([img_np.squeeze()] * 3, axis=2)
        img_np = img_np.astype(np.uint8)
        path = os.path.join(out_dir, f'{prefix}_{label}.png')
        Image.fromarray(img_np).save(path)


def main():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    vis_dir = os.path.join('outputs', 'visuals_nir', opt.name)

    use_amp = len(opt.gpu_ids) > 0 and hasattr(model, 'optimize_parameters_amp')
    scaler = torch.amp.GradScaler('cuda', init_scale=2**14) if use_amp else None
    if use_amp:
        print('AMP enabled for model [%s]' % model.name())

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        if hasattr(model, 'update_attn_scale'):
            model.update_attn_scale(epoch, opt.niter, opt.niter_decay)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)

            if use_amp:
                model.optimize_parameters_amp(scaler)
            else:
                model.optimize_parameters()

            # ── periodic visual save ───────────────────────────────────────
            if total_steps % VIS_FREQ == 0:
                visuals = model.get_current_visuals()
                prefix = f'ep{epoch:03d}_step{total_steps:07d}'
                save_visuals(visuals, vis_dir, prefix)

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
