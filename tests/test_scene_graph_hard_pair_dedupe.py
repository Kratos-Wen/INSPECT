from __future__ import annotations

import numpy as np

from inspect_assist.__main__ import _load_runtime_package

_load_runtime_package()

from inspect_runtime.components.scene_graph import GeometryAwareSceneGraphBuilder
from inspect_runtime.core_types import Detection, GeometryFrame


def _det(name: str, box: tuple[float, float, float, float], confidence: float, safe: bool) -> Detection:
    return Detection(name=name, xyxy=box, confidence=confidence, meta={"identity_safe": safe})


def test_overlapping_hard_pair_hypotheses_are_one_scene_node() -> None:
    builder = GeometryAwareSceneGraphBuilder(hard_pair_dedupe_iou=0.8)
    geometry = GeometryFrame(depth=np.ones((100, 100), dtype=np.float32))
    detections = [
        _det("type_5_gearbox_housing", (10.0, 10.0, 70.0, 70.0), 0.49, False),
        _det("type_6_gearbox_housing", (10.5, 10.5, 70.5, 70.5), 0.72, True),
        _det("type_7_gear", (35.0, 35.0, 50.0, 50.0), 0.55, True),
    ]

    graph = builder.build(detections, geometry, relevant_detections=detections, nearest_index=1)

    assert graph.stats["collapsed_hard_pair_nodes"] == 1.0
    assert all(
        not ({relation.subject_name, relation.object_name} == {"type_5_gearbox_housing", "type_6_gearbox_housing"})
        for relation in graph.relations
    )
    assert any("housing" in relation.subject_name or "housing" in relation.object_name for relation in graph.relations)
