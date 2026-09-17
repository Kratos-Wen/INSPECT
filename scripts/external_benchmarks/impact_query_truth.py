"""Resolve evaluation-only IMPACT world truth at the observed instant."""

from bisect import bisect_right


def world_state_at(annotation, frame, component):
    sequence = annotation["state_sequence"]
    frames = [int(row["frame"]) for row in sequence]
    if frames != sorted(frames):
        raise ValueError("World-state sequence must be sorted by frame")
    if int(frame) < 0:
        raise ValueError("Query frame must be nonnegative")
    index = bisect_right(frames, int(frame)) - 1
    state = annotation["initial_state_vector"] if index < 0 else sequence[index]["state"]
    value = int(state[int(component)])
    if value not in (-1, 0, 1):
        raise ValueError(f"Unknown IMPACT world state: {value}")
    return value


def relabel_query(row, annotation):
    if not row["claim_id"].endswith(".installed_correctly"):
        raise ValueError(f"Unsupported claim semantics: {row['claim_id']}")
    value = world_state_at(annotation, row["reference_query_frame"], row["component_id"])
    truth = "supported" if value == 1 else "contradicted"
    updated = dict(row)
    updated.setdefault("source_event_world_value", row["world_value"])
    updated.setdefault("source_event_semantic_outcome", row["semantic_outcome"])
    updated.update(world_value=value, semantic_outcome=truth,
                   truth_time_basis="front ASR at reference_query_frame")
    if "claim_prediction" in row:
        updated["utility"] = (2 if row["claim_prediction"] == truth
                              else int(row["raw_claim_decision"] == truth))
    return updated
