"""Compatibility entrypoint for older in-flight YOLO training jobs.

The active TwinSwap training path is ``training.twinswap_yolo_trainer``.
This module is kept only so Windows multiprocessing workers from older
``python -m training.train_existing_twinswap_dataset`` runs can import their
original main module while shutting down or restarting dataloaders.
"""

from __future__ import annotations

from .twinswap_yolo_trainer import main


if __name__ == "__main__":
    main()
