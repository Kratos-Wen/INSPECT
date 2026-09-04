"""Training utilities for INSPECT perception models."""

__all__ = [
    "TwinSwapConfig",
    "TwinSwapDetectionTrainer",
    "TwinSwapDatasetBuilder",
    "TwinSwapTrainingRunner",
    "NuisanceMiner",
    "NuisanceMiningResult",
    "main",
]


def __getattr__(name: str):
    if name in __all__:
        from . import twinswap_yolo_trainer

        return getattr(twinswap_yolo_trainer, name)
    raise AttributeError(name)
