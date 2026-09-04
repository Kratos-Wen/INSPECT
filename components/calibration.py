"""Optional expert-score calibration components."""

from __future__ import annotations

from ..core_types import StepPrediction


class IdentityStepCalibrator:
    """Leave expert scores unchanged."""

    def calibrate(self, prediction: StepPrediction) -> StepPrediction:
        return prediction
