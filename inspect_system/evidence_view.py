"""Generic evidence-view ontology for claim-disambiguating inspection.

INSPECT does not name object-specific viewpoints such as "left side of the
gear".  It maps missing procedural evidence to reusable evidence-view
requirements, then grounds those requirements into the robot's calibrated
view lattice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class EvidenceViewType(str, Enum):
    """Reusable observation condition needed to verify a class of evidence."""

    OBJECT_PRESENCE = "object_presence_view"
    IDENTITY_DISAMBIGUATION = "identity_disambiguation_view"
    FINE_GEOMETRY = "fine_geometry_view"
    SURFACE_DETAIL = "surface_detail_view"
    ORIENTATION_MARKER = "orientation_marker_view"
    COMPLETENESS = "completeness_view"
    CONTACT_VERIFICATION = "contact_verification_view"
    BOUNDARY_ALIGNMENT = "boundary_alignment_view"
    GAP_VISIBILITY = "gap_visibility_view"
    CONTAINMENT_VERIFICATION = "containment_verification_view"
    INSERTION_VERIFICATION = "insertion_verification_view"
    SLOT_RELATION = "slot_relation_view"
    RELATIVE_POSITION = "relative_position_view"
    OCCLUSION_RECOVERY = "occlusion_recovery_view"
    GLARE_REDUCTION = "glare_reduction_view"
    BLUR_RECOVERY = "blur_recovery_view"
    SCALE_CONTEXT = "scale_context_view"
    TARGET_CENTERING = "target_centering_view"
    LOW_CONFIDENCE_RECOVERY = "low_confidence_recovery_view"
    CLAIM_DISAMBIGUATION = "claim_disambiguation_view"


@dataclass(frozen=True)
class EvidenceGap:
    """A missing procedural evidence cue that should be revealed by inspection."""

    key: str
    evidence_type: str
    importance: float = 1.0
    target_object: Optional[str] = None
    slot_object: Optional[str] = None
    relation: Optional[str] = None


@dataclass(frozen=True)
class EvidenceViewRequirement:
    """A generic view requirement derived from one missing evidence cue."""

    view_type: EvidenceViewType
    evidence_key: str
    importance: float = 1.0
    target_object: Optional[str] = None
    slot_object: Optional[str] = None
    relation: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["view_type"] = self.view_type.value
        return payload


def evidence_view_requirements_from_gaps(
    gaps: Sequence[EvidenceGap],
    features: Optional[Mapping[str, float]] = None,
) -> List[EvidenceViewRequirement]:
    """Map dataset-specific evidence gaps to reusable evidence-view types."""

    features = dict(features or {})
    requirements: List[EvidenceViewRequirement] = []
    seen: set[Tuple[str, str, str, str]] = set()

    def add(gap: EvidenceGap, view_type: EvidenceViewType, weight: float = 1.0, rationale: str = "") -> None:
        key = (
            view_type.value,
            str(gap.key),
            str(gap.target_object or ""),
            str(gap.slot_object or ""),
        )
        if key in seen:
            return
        seen.add(key)
        requirements.append(
            EvidenceViewRequirement(
                view_type=view_type,
                evidence_key=str(gap.key),
                importance=max(0.0, float(gap.importance) * float(weight)),
                target_object=gap.target_object,
                slot_object=gap.slot_object,
                relation=gap.relation,
                rationale=rationale or f"{gap.evidence_type} requires {view_type.value}",
            )
        )

    for gap in gaps:
        kind = str(gap.evidence_type or "generic")
        relation = str(gap.relation or "").lower()
        if kind == "object_identity":
            add(gap, EvidenceViewType.IDENTITY_DISAMBIGUATION, 1.0, "separate target identity from plausible hard counterparts")
            add(gap, EvidenceViewType.FINE_GEOMETRY, 0.70, "fine local geometry is needed for hard-pair identity")
        elif kind == "detector_uncertainty":
            add(gap, EvidenceViewType.LOW_CONFIDENCE_RECOVERY, 1.0, "recover uncertain detector evidence")
            add(gap, EvidenceViewType.CLAIM_DISAMBIGUATION, 0.60, "reduce claim ambiguity")
        elif kind == "object_presence":
            add(gap, EvidenceViewType.OBJECT_PRESENCE, 1.0, "make the target object visibly present")
            if features.get("target_confidence", 1.0) < 0.40:
                add(gap, EvidenceViewType.LOW_CONFIDENCE_RECOVERY, 0.55, "object evidence is low-confidence")
        elif kind == "target_centering":
            add(gap, EvidenceViewType.TARGET_CENTERING, 1.0, "move the target into a better observation region")
        elif kind == "hand_occlusion":
            add(gap, EvidenceViewType.OCCLUSION_RECOVERY, 1.0, "recover evidence hidden by an occluder")
        elif kind == "slot_presence":
            add(gap, EvidenceViewType.SLOT_RELATION, 1.0, "make the relevant slot visible")
            add(gap, EvidenceViewType.SCALE_CONTEXT, 0.45, "see target and slot in the same context")
        elif kind == "object_slot_alignment":
            add(gap, EvidenceViewType.BOUNDARY_ALIGNMENT, 1.0, "verify alignment boundary")
            add(gap, EvidenceViewType.GAP_VISIBILITY, 0.65, "make gap or misalignment visible")
            add(gap, EvidenceViewType.SLOT_RELATION, 0.60, "ground alignment to the target slot")
        elif kind == "contact_relation":
            add(gap, EvidenceViewType.CONTACT_VERIFICATION, 1.0, "verify physical contact")
            add(gap, EvidenceViewType.GAP_VISIBILITY, 0.70, "distinguish contact from a small gap")
        elif kind == "insertion_relation":
            add(gap, EvidenceViewType.INSERTION_VERIFICATION, 1.0, "verify insertion postcondition")
            add(gap, EvidenceViewType.CONTAINMENT_VERIFICATION, 0.85, "distinguish inside from near or overlapping")
            add(gap, EvidenceViewType.SLOT_RELATION, 0.65, "see target and receiving slot together")
            add(gap, EvidenceViewType.GAP_VISIBILITY, 0.55, "distinguish fully seated from a small visible gap")
            add(gap, EvidenceViewType.BOUNDARY_ALIGNMENT, 0.45, "verify the target boundary against the receiving part")
        elif kind == "inside_cavity_visibility" or relation in {"inside", "in", "inserted", "seated"}:
            add(gap, EvidenceViewType.CONTAINMENT_VERIFICATION, 1.0, "make containment evidence visible")
            add(gap, EvidenceViewType.SLOT_RELATION, 0.60, "observe target relative to cavity or slot")
        elif kind == "orientation_marker":
            add(gap, EvidenceViewType.ORIENTATION_MARKER, 1.0, "make orientation marker or tooth direction visible")
            add(gap, EvidenceViewType.FINE_GEOMETRY, 0.60, "recover fine orientation detail")
        elif kind == "screw_hole_visibility":
            add(gap, EvidenceViewType.FINE_GEOMETRY, 0.85, "make small hole geometry visible")
            add(gap, EvidenceViewType.SURFACE_DETAIL, 0.60, "recover local surface detail")
        else:
            add(gap, EvidenceViewType.CLAIM_DISAMBIGUATION, 1.0, "generic claim ambiguity")

    if not requirements and features.get("verification_confidence", 1.0) < 0.5:
        fallback = EvidenceGap(key="visual:claim_uncertain", evidence_type="generic", importance=1.0)
        add(fallback, EvidenceViewType.CLAIM_DISAMBIGUATION, 1.0, "verification confidence is low")

    return requirements
