"""Load the final CORAL-adapted student model and related utilities.

Standalone version: every module this resolves lives inside `final_student/`.
Nothing is imported from `student/`, `khushi/student/`, or `domain_adapt/`.

Model chain actually instantiated (see `load_student_model`):

    CoralStudentModel                      final_student/coral/coral_model.py
      -> Pix2Pix_attn_Student_FSP_FFL_Model  final_student/coral/pix2pix_attn_student_fsp_ffl_model.py
        -> Pix2Pix_attn_Student_FSP_Model    final_student/models/pix2pix_attn_student_fsp_model_changed.py
          -> BaseModel                       final_student/models/base_model.py
    networks                                 final_student/models/networks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Package root — makes `models`, `util`, `pytorch_ssim`, `pipeline`, `methods`
# importable, and `coral/` holds flat modules imported by bare name.
_PKG_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_PKG_ROOT / "coral"), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Repo root — only used to default the teacher checkpoint location.
_REPO_ROOT = _PKG_ROOT.parent


class ModelOpt:
    """Inference-time options for the CORAL student.

    CORAL training (`coral/train.py`) never wrote an `opt.pkl`, so these must be
    reconstructed by hand and must match `train.py`'s defaults exactly. The
    architecture fields in particular: `_load_from_dir` uses `strict=False`, so a
    mismatch loads silently with missing keys and produces noise rather than raising.

    Note `which_model_netA = 'resnet_9blocks'` — netA is the one module that is NOT
    unet_256, and getting it wrong is the easiest way to silently break this.
    """

    def __init__(self, ckpt_dir: Path):
        self.isTrain           = False
        self.gpu_ids           = [0] if torch.cuda.is_available() else []
        self.checkpoints_dir   = str(ckpt_dir.parent)
        self.name              = ckpt_dir.name
        self.which_epoch       = "latest"
        self.batchSize         = 1
        self.fineSize          = 256
        self.input_nc          = 3
        self.output_nc         = 3
        self.ngf               = 64
        self.ndf               = 64
        self.which_model_netG  = "unet_256"
        self.which_model_netA  = "resnet_9blocks"
        self.which_model_netG2 = "unet_256"   # netH reads this key
        self.norm              = "instance"
        self.no_dropout        = True
        self.init_type         = "normal"
        self.dataset_mode      = "unaligned"
        self.which_direction   = "AtoB"

        # Distillation / teacher fields. Unused at inference (the frozen teacher is
        # only built when isTrain), but the model class reads them during initialize().
        self.teacher_checkpoints_dir = str(
            _REPO_ROOT / "AttentionGAN-for-Cloud-removal" / "outputs" / "checkpoints"
        )
        self.teacher_name       = "sen12mscr_nir_teacher_v4c"
        self.teacher_which_epoch = "latest"
        self.gamma_hall         = 1.0
        self.lambda3            = 1.0
        self.lambda_A           = 100.0
        self.CORAL_WEIGHT       = 1.0
        self.device = torch.device(
            f"cuda:{self.gpu_ids[0]}" if self.gpu_ids else "cpu"
        )


def load_student_model(weights_path: str):
    """Load the final CORAL student and return (netG, netA, netH, device).

    `weights_path` names a file but is used as a directory + epoch tag:
    the parent dir becomes `--name`/`--checkpoints_dir`, and `which_epoch` is
    the filename prefix — so `latest_net_G.pth` -> epoch `latest`,
    `epoch_15_net_G.pth` -> epoch `epoch`. netG / netA / netH are then all loaded
    by the model wrapper; the single `_net_G` filename is just a handle.
    """
    weights_path = Path(weights_path)
    ckpt_dir = weights_path.parent

    opt = ModelOpt(ckpt_dir)
    opt.which_epoch = weights_path.name.split("_")[0]

    from coral_model import CoralStudentModel

    model = CoralStudentModel()
    model.initialize(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netG = model.netG.to(device)
    netA = model.netA.to(device)
    netH = model.netH.to(device)
    return netG, netA, netH, device


def apply_ttbn(net: nn.Module) -> None:
    """Test-Time Batch Normalization: use per-batch stats instead of running stats.

    The student was trained on SEN12MS-CR whose norm-layer running stats don't match
    LISS-4. Switching norm layers to train mode with track_running_stats=False makes
    them compute statistics from each batch, adapting on the fly.

    IMPORTANT: this student's norm='instance' builds nn.InstanceNorm2d(affine=False,
    track_running_stats=True) (see get_norm_layer in networks.py) -- NOT the PyTorch
    default (track_running_stats=False). With track_running_stats=True, InstanceNorm2d
    behaves like BatchNorm2d at eval time: it uses accumulated running stats instead of
    per-instance statistics. So these layers must be treated the same as BatchNorm2d
    here, or the model silently keeps using SEN12MS-CR-domain running stats on LISS-4
    input, which is what was producing near-black inference outputs.
    """
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            m.train()
            m.track_running_stats = False


def run_adabn(
    netG: nn.Module,
    netA: nn.Module,
    netH: nn.Module,
    loader,
    device: torch.device,
    n_batches: int = 50,
) -> None:
    """AdaBN: update BatchNorm running statistics using LISS-4 inputs.

    Runs a full forward pass (no gradients, no output accumulation) for
    n_batches batches drawn from the already-adapted LISS-4 loader.
    Norm layers accumulate running mean/var via cumulative moving average
    (momentum=None -> alpha=1/count), then return to eval mode with those
    updated statistics. Weights are untouched.

    Only BatchNorm2d layers are affected. The student's norm='instance' builds
    nn.InstanceNorm2d(affine=False, track_running_stats=True) — because
    track_running_stats=True, these layers use accumulated running statistics at
    eval() time exactly like BatchNorm2d instead of per-instance statistics, which
    *is* a real domain-mismatch bug. But cumulative-moving-average
    reset+reaccumulation (this function's approach) turned out to be numerically
    unstable for InstanceNorm2d in practice (produced literal all-zero output on real
    scenes). The fix used instead is in apply_ttbn: permanently setting
    track_running_stats=False on InstanceNorm2d makes it always use per-instance
    statistics, in any train/eval mode — which is exactly what happens during training
    too (see _apply_instance_norm: use_input_stats = self.training or not
    self.track_running_stats), so it reproduces training-time behavior exactly and
    needs no accumulation step. This function is therefore a near-no-op for the
    InstanceNorm-only student and is kept for any real BatchNorm2d layers.
    """
    def _set_bn_adabn(net: nn.Module, active: bool) -> None:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                if active:
                    m.train()
                    m.reset_running_stats()
                    m.track_running_stats = True
                    m.momentum = None   # cumulative moving average: alpha = 1/count
                else:
                    m.eval()
                    m.track_running_stats = True
                    m.momentum = 0.1    # restore default for future use

    for net in (netG, netA, netH):
        _set_bn_adabn(net, active=True)

    i = -1
    with torch.no_grad():
        for i, (tensors, _ys, _xs) in enumerate(loader):
            if i >= n_batches:
                break
            adapted = tensors.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                fake_c = netH(adapted)
                _      = netA(adapted)
                _      = netG(torch.cat([adapted, fake_c], dim=1))

    for net in (netG, netA, netH):
        _set_bn_adabn(net, active=False)

    print(f"  AdaBN: updated BN stats over {min(n_batches, i + 1)} batches")


def try_compile(*nets: nn.Module):
    """Attempt torch.compile on each network; silently skip if unavailable."""
    compiled = []
    for net in nets:
        try:
            compiled.append(torch.compile(net))
        except Exception as e:
            print(f"torch.compile skipped ({e})")
            compiled.append(net)
    return compiled
