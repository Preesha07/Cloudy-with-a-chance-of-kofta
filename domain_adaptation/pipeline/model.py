"""Load the student model and related utilities."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn

_PKG_ROOT  = Path(__file__).resolve().parents[1]   # domain_adaptation/
_REPO_ROOT = Path(__file__).resolve().parents[2]   # repo root

# The student model code is vendored under domain_adaptation/student/ so this tree
# is self-contained.  Two entries are needed, mirroring how the student repo is laid
# out: `student/` itself (its model files do `import util.util`, `import pytorch_ssim`)
# and its parent (for `from student.models... import ...`).
#
# Order matters: _PKG_ROOT must precede _REPO_ROOT on sys.path, or the repo-root
# `student/` tree shadows the vendored one and this package silently stops being
# self-contained.
_STUDENT_ROOT = _PKG_ROOT / "student"
for _p in (str(_REPO_ROOT), str(_PKG_ROOT), str(_STUDENT_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


class MockOpt:
    """Fallback model options when opt.pkl is absent next to the checkpoint."""
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
        self.which_model_netA  = "unet_256"
        self.which_model_netG2 = "unet_256"   # netH reads this key
        self.norm              = "instance"
        self.no_dropout        = True
        self.init_type         = "normal"
        self.dataset_mode      = "unaligned"
        self.which_direction   = "AtoB"


def _build_opt(weights_path: Path) -> object:
    ckpt_dir = weights_path.parent
    opt_pkl = ckpt_dir / "opt.pkl"
    if opt_pkl.exists():
        with open(opt_pkl, "rb") as f:
            opt = pickle.load(f)
        opt.isTrain = False
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
        # Do NOT override opt.norm — preserve what training used (instance for all student runs).
        print(f"Loaded opt from {opt_pkl}  [norm={getattr(opt, 'norm', '?')}]")
    else:
        opt = MockOpt(ckpt_dir)
        print(f"opt.pkl not found — using MockOpt defaults (unet_256 / instance norm)")
    opt.which_epoch = weights_path.name.split("_")[0]
    return opt


def load_student_model(weights_path: str, model_variant: str = "base"):
    """Load the student model and return (netG, netA, netH, device).

    model_variant:
        'base'  → Pix2Pix_attn_Student_Model        (student/models/)
        'fsp'   → Pix2Pix_attn_Student_FSP_Model    (student/models/)
        'coral' → CoralStudentModel                 (methods/coral/coral_model.py)
    all resolved inside domain_adaptation/, not from the repo-root student trees.
    """
    weights_path = Path(weights_path)

    if model_variant == "coral":
        # CORAL training never saved opt.pkl; reconstruct opt to match train_coral.py defaults.
        ckpt_dir = weights_path.parent
        opt = MockOpt(ckpt_dir)
        opt.which_model_netA  = "resnet_9blocks"
        opt.norm              = "instance"
        opt.which_epoch       = weights_path.name.split("_")[0]
        opt.isTrain           = False
        opt.teacher_checkpoints_dir = str(
            _REPO_ROOT / "AttentionGAN-for-Cloud-removal" / "outputs" / "checkpoints"
        )
        opt.teacher_name      = "sen12mscr_nir_teacher_v4c"
        opt.teacher_which_epoch = "latest"
        opt.gamma_hall        = 1.0
        opt.lambda3           = 1.0
        opt.lambda_A          = 100.0
        opt.CORAL_WEIGHT      = 1.0
        opt.device            = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu")

        # coral_model.py imports its siblings flat (`from coral_loss import ...`),
        # so the coral dir itself has to be importable.
        _coral_dir = str(_PKG_ROOT / "methods" / "coral")
        if _coral_dir not in sys.path:
            sys.path.insert(0, _coral_dir)

        from coral_model import CoralStudentModel as ModelClass
    else:
        opt = _build_opt(weights_path)

    if model_variant == "fsp":
        from student.models.pix2pix_attn_student_fsp_model import (
            Pix2Pix_attn_Student_FSP_Model as ModelClass,
        )
    elif model_variant != "coral":
        from student.models.pix2pix_attn_student_model import (
            Pix2Pix_attn_Student_Model as ModelClass,
        )

    model = ModelClass()
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
    """AdaBN: update BatchNorm/InstanceNorm running statistics using LISS-4 inputs.

    Runs a full forward pass (no gradients, no output accumulation) for
    n_batches batches drawn from the already-adapted LISS-4 loader.
    Norm layers accumulate running mean/var via cumulative moving average
    (momentum=None → alpha=1/count), then return to eval mode with those
    updated statistics.  Weights are untouched.

    Only BatchNorm2d layers are affected.  The student's norm='instance' builds
    nn.InstanceNorm2d(affine=False, track_running_stats=True) (see get_norm_layer in
    networks.py) — because track_running_stats=True, these layers use accumulated
    running statistics at eval() time exactly like BatchNorm2d instead of per-instance
    statistics, which *is* a real domain-mismatch bug (SEN12MS-CR-domain running stats
    applied to LISS-4 input).  But cumulative-moving-average reset+reaccumulation (this
    function's approach) turned out to be numerically unstable for InstanceNorm2d in
    practice (produced literal all-zero output on real scenes).  The fix used instead is
    in apply_ttbn: permanently setting track_running_stats=False on InstanceNorm2d makes
    it always use per-instance statistics, in any train/eval mode — which is exactly what
    happens during training too (see _apply_instance_norm: use_input_stats = self.training
    or not self.track_running_stats), so it reproduces training-time behavior exactly and
    needs no accumulation step.  This function is kept as a true no-op for the standard
    student (InstanceNorm-only) and remains live for models that do use real BatchNorm2d
    (e.g. the coral variant).
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
