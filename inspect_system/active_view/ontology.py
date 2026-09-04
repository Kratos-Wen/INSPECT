"""Small reusable evidence ontology for the Gearbox INSPECT experiments."""

from __future__ import annotations

from typing import Dict, Mapping


CLAIM_ALIAS: Mapping[str, str] = {
    "step2_small_gear_inserted": "gear_inserted",
    "step3_big_gear_inserted": "gear_inserted",
    "step4_cover_seated": "cover_seated",
    "small_gear_inserted": "gear_inserted",
    "big_gear_inserted": "gear_inserted",
    "cover_fully_seated": "cover_seated"
}


CLAIM_ROLES: Dict[str, Dict[str, float]] = {
    "gear_inserted": {
        "identity_disambiguation_view": 1.00,
        "insertion_verification_view": 1.00,
        "containment_verification_view": 0.85,
        "slot_relation_view": 0.70,
        "claim_disambiguation_view": 0.55,
    },
    "cover_seated": {
        "gap_visibility_view": 1.00,
        "boundary_alignment_view": 0.85,
        "contact_verification_view": 0.65,
        "claim_disambiguation_view": 0.50,
    },
    "cover_aligned": {
        "boundary_alignment_view": 1.00,
        "gap_visibility_view": 0.50,
        "claim_disambiguation_view": 0.45,
    },
    "object_identity": {
        "identity_disambiguation_view": 1.00,
        "claim_disambiguation_view": 0.65,
    },
    "state_validity": {
        "claim_disambiguation_view": 1.00,
    },
}


REASON_TO_ROLE: Mapping[str, str] = {
    "hand_occlusion": "occlusion_recovery_view",
    "object_occlusion": "occlusion_recovery_view",
    "occlusion": "occlusion_recovery_view",
    "rotation": "claim_disambiguation_view",
    "bad_angle": "claim_disambiguation_view",
    "view_angle": "claim_disambiguation_view",
    "identity_unclear": "identity_disambiguation_view",
    "gap_unclear": "gap_visibility_view",
    "alignment_unclear": "boundary_alignment_view",
    "insertion_unclear": "insertion_verification_view",
}


def normalize_key(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def normalize_claim(claim_id: object, fallback: str = "state_validity") -> str:
    claim = normalize_key(claim_id)
    return CLAIM_ALIAS.get(claim, claim or fallback)


def roles_for_claim(claim_id: object) -> Dict[str, float]:
    claim = normalize_claim(claim_id)
    if claim in CLAIM_ROLES:
        return dict(CLAIM_ROLES[claim])
    return dict(CLAIM_ROLES["state_validity"])


def counterfactual_family_for_role(
    *,
    claim_id: object = "",
    evidence_role: object = "",
    role_projection: object = "",
    declared_family: object = "",
) -> str:
    """Map an evidence role to the claim-alternative family it separates."""

    declared = normalize_key(declared_family)
    if declared and declared not in {"generic", "unknown", "none"}:
        return declared

    projection = normalize_key(role_projection)
    role = normalize_key(evidence_role)
    claim = normalize_claim(claim_id)
    if role == "identity_disambiguation_view":
        return "identity"
    if role in {
        "insertion_verification_view",
        "containment_verification_view",
        "slot_relation_view",
    }:
        return "spatial_relation"
    if role in {
        "gap_visibility_view",
        "boundary_alignment_view",
        "contact_verification_view",
    }:
        return "seating_contact"
    if role == "claim_disambiguation_view":
        if projection == "identity" or claim == "object_identity":
            return "identity"
        if "cover" in claim:
            return "seating_contact"
        if "gear" in claim:
            return "spatial_relation"
    if "cover" in claim:
        return "seating_contact"
    return "generic"


def infer_role_from_fields(
    *,
    claim_id: object = "",
    evidence_view_type: object = "",
    reason: object = "",
    missing_evidence: object = None,
    revealed_evidence: object = None,
) -> str:
    explicit = normalize_key(evidence_view_type)
    if explicit:
        return explicit
    reason_key = normalize_key(reason)
    if reason_key in REASON_TO_ROLE:
        return REASON_TO_ROLE[reason_key]
    evidence_blob = " ".join(str(v).lower() for v in [missing_evidence, revealed_evidence] if v)
    if "gap" in evidence_blob or "seated" in evidence_blob:
        return "gap_visibility_view"
    if "align" in evidence_blob or "boundary" in evidence_blob:
        return "boundary_alignment_view"
    if "insert" in evidence_blob or "inside" in evidence_blob or "contain" in evidence_blob:
        return "insertion_verification_view"
    if "identity" in evidence_blob or "class" in evidence_blob or "gear" in evidence_blob:
        return "identity_disambiguation_view"
    roles = roles_for_claim(claim_id)
    return next(iter(roles.keys()), "claim_disambiguation_view")
