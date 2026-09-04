"""Session-local spherical evidence field for causal online view updates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

from .ontology import normalize_claim, normalize_key
from .view_lattice import ViewNode


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class SessionRayAtom:
    session_id: str
    claim_id: str
    evidence_role: str
    counterfactual_family: str
    yaw: float
    elevation: float
    signed_gain: float
    weight: float


@dataclass
class SessionRayEvidenceField:
    """Bounded posterior over evidence gain at destination viewing rays.

    Updates use only the selected observation after frozen re-verification.
    Atoms are isolated by session and therefore cannot transfer robot-test
    outcomes across trials or assemblies.
    """

    strength: float = 1.0
    prior_strength: float = 1.0
    gain_scale: float = 1.0
    yaw_bandwidth: float = math.radians(45.0)
    elevation_bandwidth: float = math.radians(30.0)
    max_atoms: int = 512
    atoms: list[SessionRayAtom] = field(default_factory=list)

    def update(
        self,
        *,
        session_id: str,
        claim_id: str,
        evidence_role: str,
        counterfactual_family: str,
        view_id: str,
        views: Mapping[str, ViewNode],
        signed_gain: float,
        weight: float,
    ) -> bool:
        session = str(session_id).strip()
        if (
            not session
            or view_id not in views
            or not math.isfinite(float(signed_gain))
            or float(weight) <= 0.0
        ):
            return False
        node = views[view_id]
        self.atoms.append(
            SessionRayAtom(
                session_id=session,
                claim_id=normalize_claim(claim_id),
                evidence_role=normalize_key(evidence_role),
                counterfactual_family=normalize_key(counterfactual_family),
                yaw=math.radians(float(node.yaw)),
                elevation=math.radians(float(node.elevation)),
                signed_gain=max(-1.0, min(1.0, float(signed_gain))),
                weight=max(0.0, float(weight)),
            )
        )
        if len(self.atoms) > max(1, int(self.max_atoms)):
            del self.atoms[: len(self.atoms) - int(self.max_atoms)]
        return True

    def factor(
        self,
        *,
        session_id: str | None,
        claim_id: str,
        evidence_role: str,
        counterfactual_weights: Mapping[str, float],
        candidate_view: str,
        views: Mapping[str, ViewNode],
    ) -> float:
        session = str(session_id or "").strip()
        if not session or candidate_view not in views or self.strength <= 0.0:
            return 1.0
        claim = normalize_claim(claim_id)
        role = normalize_key(evidence_role)
        families = {
            normalize_key(key): max(0.0, float(value))
            for key, value in counterfactual_weights.items()
            if normalize_key(key) and float(value) > 0.0
        }
        if not families:
            families = {"generic": 1.0}
        node = views[candidate_view]
        yaw = math.radians(float(node.yaw))
        elevation = math.radians(float(node.elevation))
        weighted_gain = 0.0
        support = 0.0
        yaw_bandwidth = max(1e-6, float(self.yaw_bandwidth))
        elevation_bandwidth = max(1e-6, float(self.elevation_bandwidth))
        for atom in self.atoms:
            if (
                atom.session_id != session
                or atom.claim_id != claim
                or atom.evidence_role != role
                or atom.counterfactual_family not in families
            ):
                continue
            distance = (_wrap_angle(yaw - atom.yaw) / yaw_bandwidth) ** 2 + (
                (elevation - atom.elevation) / elevation_bandwidth
            ) ** 2
            similarity = math.exp(-0.5 * distance)
            mass = similarity * atom.weight * families[atom.counterfactual_family]
            weighted_gain += mass * atom.signed_gain
            support += mass
        if support <= 1e-9:
            return 1.0
        mean_gain = weighted_gain / support
        reliability = support / (support + max(1e-6, float(self.prior_strength)))
        exponent = (
            max(0.0, float(self.strength))
            * reliability
            * mean_gain
            / max(1e-6, float(self.gain_scale))
        )
        return min(2.0, max(0.5, math.exp(exponent)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strength": self.strength,
            "prior_strength": self.prior_strength,
            "gain_scale": self.gain_scale,
            "yaw_bandwidth": self.yaw_bandwidth,
            "elevation_bandwidth": self.elevation_bandwidth,
            "max_atoms": self.max_atoms,
            "atoms": [asdict(atom) for atom in self.atoms],
            "uses_candidate_view_images": False,
            "uses_robot_utility_labels": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionRayEvidenceField":
        field_model = cls(
            strength=max(0.0, float(payload.get("strength", 1.0))),
            prior_strength=max(1e-6, float(payload.get("prior_strength", 1.0))),
            gain_scale=max(1e-6, float(payload.get("gain_scale", 1.0))),
            yaw_bandwidth=max(
                1e-6,
                float(payload.get("yaw_bandwidth", math.radians(45.0))),
            ),
            elevation_bandwidth=max(
                1e-6,
                float(payload.get("elevation_bandwidth", math.radians(30.0))),
            ),
            max_atoms=max(1, int(payload.get("max_atoms", 512))),
        )
        field_model.atoms = [
            SessionRayAtom(
                session_id=str(raw.get("session_id", "")),
                claim_id=normalize_claim(raw.get("claim_id", "")),
                evidence_role=normalize_key(raw.get("evidence_role", "")),
                counterfactual_family=normalize_key(
                    raw.get("counterfactual_family", "generic")
                ),
                yaw=float(raw.get("yaw", 0.0)),
                elevation=float(raw.get("elevation", 0.0)),
                signed_gain=max(
                    -1.0,
                    min(1.0, float(raw.get("signed_gain", 0.0))),
                ),
                weight=max(0.0, float(raw.get("weight", 0.0))),
            )
            for raw in payload.get("atoms", [])
            if isinstance(raw, Mapping) and str(raw.get("session_id", "")).strip()
        ][-field_model.max_atoms :]
        return field_model
