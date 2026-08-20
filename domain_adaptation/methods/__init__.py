from .dataset_norm import DatasetNormAdapter

REGISTRY: dict = {
    "dataset_norm": DatasetNormAdapter,
}


def get_adapter(name: str, **kwargs):
    if name not in REGISTRY:
        raise ValueError(f"Unknown adaptation method '{name}'. Choose from: {list(REGISTRY)}")
    cls = REGISTRY[name]
    import inspect
    # Only pass kwargs the adapter's __init__ actually accepts
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid and v is not None}
    return cls(**filtered)
