"""Dataset normalisation: no preprocessing — adapts via test-time batch normalisation."""
from .base import BaseAdapter


class DatasetNormAdapter(BaseAdapter):
    """Passes batches through unchanged; relies on TTBN for domain shift."""

    name = "dataset_norm"
    needs_ref_image = False
    drop_last = True   # TTBN needs full batches for stable variance estimates
