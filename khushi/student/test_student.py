"""
SAR-free inference script for the student hallucination model.

Only RG+NIR optical input (real_A) is needed at test time.
The data loader still loads real_C (SAR) from the dataset but the student
model's test() method ignores it — so the test can also be run against the
original SEN12MSCR_prepared/ (which has RGB C images) without issue.

Usage (run from AttentionGAN-for-Cloud-removal/):
    python test_student.py \\
        --dataroot        ../SEN12MSCR_student \\
        --name            sen12mscr_student_v1 \\
        --model           pix2pix_attn_student \\
        --dataset_mode    unaligned_sar \\
        --checkpoints_dir ./outputs/checkpoints \\
        --phase           val \\
        --which_epoch     latest \\
        --gpu_ids         0
"""

import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html


class StudentTestOptions(TestOptions):
    """Extends TestOptions with the minimal extra fields the student model needs
    to instantiate at test time (no teacher loading happens, but argparse must
    not fail if the fields are present in the model's initialize())."""

    def initialize(self):
        TestOptions.initialize(self)
        # These are read by StudentTrainOptions during training but are NOT used
        # at test time.  Defined here so the same model class works with both
        # train_student.py and test_student.py without AttributeError.
        self.parser.add_argument('--teacher_checkpoints_dir', type=str,
                                 default='./outputs/checkpoints')
        self.parser.add_argument('--teacher_name', type=str,
                                 default='sen12mscr_teacher_v1')
        self.parser.add_argument('--teacher_which_epoch', type=str,
                                 default='latest')
        self.parser.add_argument('--gamma_hall', type=float, default=100.0)
        self.parser.add_argument('--lambda3',    type=float, default=1.0)
        self.parser.add_argument('--hook_layers', type=str,  default='3,4,5,6,7')


opt = StudentTestOptions().parse()
opt.nThreads    = 1
opt.batchSize   = 1
opt.serial_batches = True
opt.no_flip     = True

data_loader = CreateDataLoader(opt)
dataset     = data_loader.load_data()
model       = create_model(opt)
visualizer  = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))
print('name:',    opt.name)
print('web dir:', web_dir)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals  = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path[0])

webpage.save()
