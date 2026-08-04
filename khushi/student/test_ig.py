import time
import os
import torch
import torch.utils.data
import sys
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataset
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm


def evaluate_teacher_sar_dependence(model, dataloader, device):
    """
    Evaluates SAR vs Optical contribution percentages across a dataset
    for the pix2pix_attn model (Pix2Pix_attn_Model / sen12mscr_teacher_v1).

    Full forward pass:
        att_A  = netA(optical)
        fake_C = netG2(SAR)              # SAR → optical intermediate
        g_B    = netG(cat[optical, fake_C])
        fake_B = g_B * att_A + optical * (1 − att_A)

    SAR attribution: interp_sar → netG2 → fake_C → netG → g_B → fake_B
    Optical attribution: interp_opt → netA, netG (direct), and passthrough (1−att_A)
    """
    for net in [model.netG, model.netG2, model.netA]:
        net.eval()
    model.set_requires_grad([model.netG, model.netG2, model.netA], requires_grad=False)

    total_opt_perc = 0.0
    total_sar_perc = 0.0
    count = 0

    pbar = tqdm(dataloader, desc="EG attribution", unit="batch")
    for data in pbar:
        real_A = data['A'].to(device)  # Optical
        real_C = data['C'].to(device)  # SAR

        base_opt = torch.zeros_like(real_A)
        base_sar = torch.zeros_like(real_C)

        batch_size = real_C.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)

        interp_opt = (base_opt + alpha * (real_A - base_opt)).detach().requires_grad_(True)
        interp_sar = (base_sar + alpha * (real_C - base_sar)).detach().requires_grad_(True)

        # Full pix2pix_attn forward pass
        att_A  = model.netA(interp_opt)
        fake_C = model.netG2(interp_sar)
        g_B    = model.netG(torch.cat([interp_opt, fake_C], dim=1))
        fake_B = g_B * att_A + interp_opt * (1 - att_A)

        grad_outputs = torch.ones_like(fake_B)
        grads = torch.autograd.grad(
            outputs=fake_B,
            inputs=(interp_opt, interp_sar),
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False
        )

        attr_opt = torch.sum(torch.abs((real_A - base_opt) * grads[0]))
        attr_sar = torch.sum(torch.abs((real_C - base_sar) * grads[1]))

        total_attr = attr_opt + attr_sar + 1e-8
        perc_opt = (attr_opt / total_attr).item() * 100.0
        perc_sar = (attr_sar / total_attr).item() * 100.0

        total_opt_perc += perc_opt
        total_sar_perc += perc_sar
        count += 1

        pbar.set_postfix(opt=f"{total_opt_perc/count:.1f}%", sar=f"{total_sar_perc/count:.1f}%")

    print(f"\n--- Post-Hoc Attribution Results across {count} batches ---")
    print(f"Average Optical Contribution: {total_opt_perc / count:.2f}%")
    print(f"Average SAR Contribution:     {total_sar_perc / count:.2f}%\n")


opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True         # no flip

device = torch.device(f'cuda:{opt.gpu_ids[0]}' if opt.gpu_ids else 'cpu')
num_workers = min(8, os.cpu_count() or 1)
use_pin = device.type == 'cuda'

dataset = CreateDataset(opt)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_pin,
    prefetch_factor=4 if num_workers > 0 else None,
    persistent_workers=num_workers > 0,
)
model = create_model(opt)

print(f"Running on: {device}  |  workers: {num_workers}  |  batch: {opt.batchSize}")

print("Executing post-hoc Expected Gradients evaluation...")
evaluate_teacher_sar_dependence(model, dataloader, device)
sys.exit(0)

# ---------------------------------------------------------
# The rest of the original code remains structurally intact 
# but will not be executed due to the sys.exit(0) above.
# ---------------------------------------------------------

visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
print('name:',opt.name)
print('web dir:', web_dir)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%d:process image... %s' % (i,img_path))
    #print('%d:process image... %s' % (i,img_path[1]))
    visualizer.save_images(webpage, visuals, img_path[0])
    #visualizer.save_images(webpage, visuals, img_path)
    #visualizer.save_images(webpage, visuals, img_path[1])
webpage.save()