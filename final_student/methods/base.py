"""Abstract base class for domain adaptation methods."""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class BaseAdapter(ABC):
    """Interface every domain adaptation method must implement.

    The inference loop calls:
        adapted  = adapter.adapt_input(batch, ref_batch)
        # ... model forward pass on `adapted` ...
        output   = adapter.restore_output(fake_b, batch)

    where `batch` is the original (non-adapted) tensor and `ref_batch` is the
    reference image broadcast to the same batch size.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in output directory names."""

    @property
    def needs_ref_image(self) -> bool:
        """True if this adapter requires a reference SEN12MS-CR image."""
        return False

    @property
    def drop_last(self) -> bool:
        """True → DataLoader uses drop_last=True (required for TTBN-only methods)."""
        return True

    @property
    def stretch_input(self) -> bool:
        """True → per-channel 2-98 percentile stretch of the full scene before tiling.

        Ensures the model receives input in the [0, 255] range it was trained on.
        Should be True for any method that does not otherwise remap the input range.
        """
        return False

    @property
    def scene_transform(self):
        """Optional callable applied to the merged [H, W, 3] uint8 scene array before tiling.

        Signature: (arr: np.ndarray) -> np.ndarray, both uint8.
        Return None to skip.  Applied after stretch_input so both can compose,
        though most adapters set stretch_input=False and handle everything here.
        """
        return None

    @property
    def adabn(self) -> bool:
        """True → run AdaBN (update BatchNorm running stats) before inference.

        The adapted LISS-4 loader is fed through the model with BN in train mode
        and reset running stats, then returned to eval with those updated stats.
        """
        return False

    def adapt_input(self, tensors: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Pre-process a batch of LISS-4 patches before the model forward pass.

        tensors: [B, 3, H, W] in [-1, 1], original domain
        ref:     [B, 3, H, W] in [-1, 1], SEN12MS-CR reference (may be None)
        Returns: adapted tensor in [-1, 1]
        """
        return tensors

    def restore_output(self, fake_b: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Post-process the model output to restore it to the target domain.

        fake_b:   [B, 3, H, W] in [-1, 1], model output (may have been fed adapted input)
        original: [B, 3, H, W] in [-1, 1], original (non-adapted) target-domain batch
        Returns:  restored tensor in [-1, 1]
        """
        return fake_b
