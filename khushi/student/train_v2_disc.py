"""Training driver for Experiment 1: pix2pix_attn_d2 (G2 with dedicated discriminator).

Identical to train.py except periodic visuals are written to outputs/visuals_v2/
instead of relying on the HTML visualizer (which is disabled via --no_html).
"""

import os
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util
import random

random.seed(10)

VISUAL_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'visuals_v2')

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

os.makedirs(VISUAL_DIR, exist_ok=True)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visuals = model.get_current_visuals()
            # Persist one image per visual key to outputs/visuals_v2/
            for label, image_numpy in visuals.items():
                fname = 'epoch%03d_iter%06d_%s.png' % (epoch, total_steps, label)
                util.save_image(image_numpy, os.path.join(VISUAL_DIR, fname))

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

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
