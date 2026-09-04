"""INSPECT-Active evidence-revealing view transfer.

This package learns relative evidence-revealing view changes from first-person
assistant traces and projects them onto the robot's fixed view lattice at
evaluation time. Robot fixed-view images are not used for training the policy.
"""

from .active_selector import ActiveSelector, SelectionResult
from .evidence_state import EvidenceItem, EvidenceState
from .reveal_model import PriorTableRevealModel, TrainingExample
from .transfer_gate import TransferGateDecision, TransferType
from .view_lattice import SIX_VIEWS, RelativeAction, ViewNode

__all__ = [
    "ActiveSelector",
    "EvidenceItem",
    "EvidenceState",
    "PriorTableRevealModel",
    "RelativeAction",
    "SIX_VIEWS",
    "SelectionResult",
    "TrainingExample",
    "TransferGateDecision",
    "TransferType",
    "ViewNode",
]
