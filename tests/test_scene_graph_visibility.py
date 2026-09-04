from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from inspect_runtime.components.scene_graph import GeometryAwareSceneGraphBuilder
from inspect_runtime.core_types import Detection, GeometryFrame


def _detection(name: str, box: tuple[float, float, float, float]) -> Detection:
    return Detection(name=name, xyxy=box, confidence=0.9, meta={"identity_safe": True})


def test_unreliable_depth_suppresses_3d_relations_but_preserves_2d_order() -> None:
    depth = np.ones((100, 100), dtype=np.float32)
    depth[:, 50:] = 2.0
    valid = np.ones((100, 100), dtype=bool)
    valid[20:40, 5:25] = False
    detections = [
        _detection("type_3_gear", (5.0, 20.0, 25.0, 40.0)),
        _detection("type_5_gearbox_housing", (60.0, 20.0, 80.0, 40.0)),
    ]

    graph = GeometryAwareSceneGraphBuilder(min_depth_valid_fraction=0.35).build(
        detections,
        GeometryFrame(depth=depth, valid_mask=valid),
        relevant_detections=detections,
        nearest_index=0,
    )
    predicates = {relation.predicate for relation in graph.relations}

    assert "left_of" in predicates
    assert "right_of" in predicates
    assert "in_front_of" not in predicates
    assert "behind" not in predicates
    assert graph.stats["low_depth_quality_objects"] == 1.0


def test_inner_box_median_rejects_background_depth_contamination() -> None:
    depth = np.full((100, 100), 8.0, dtype=np.float32)
    depth[18:42, 18:42] = 1.0
    depth[18:42, 68:92] = 2.0
    detections = [
        _detection("type_3_gear", (10.0, 10.0, 50.0, 50.0)),
        _detection("type_5_gearbox_housing", (60.0, 10.0, 100.0, 50.0)),
    ]

    graph = GeometryAwareSceneGraphBuilder(depth_inner_ratio=0.20).build(
        detections,
        GeometryFrame(depth=depth, valid_mask=np.ones_like(depth, dtype=bool)),
        relevant_detections=detections,
        nearest_index=0,
    )
    front = next(
        relation
        for relation in graph.relations
        if relation.subject_name == "type_3_gear" and relation.predicate == "in_front_of"
    )

    assert abs(float(front.extras["subject_depth"]) - 1.0) < 1e-6
    assert abs(float(front.extras["object_depth"]) - 2.0) < 1e-6
    assert float(front.extras["subject_depth_quality"]) > 0.9
