from __future__ import annotations

import math

from inspect_system.active_view.view_lattice import (
    SIX_VIEWS,
    action_compatibility,
)


def test_exact_relative_action_has_unit_compatibility() -> None:
    assert math.isclose(
        action_compatibility("orbit_right", "V0", "V1", SIX_VIEWS),
        1.0,
    )


def test_orthogonal_and_opposite_actions_do_not_contribute() -> None:
    assert math.isclose(
        action_compatibility("orbit_up", "V0", "V1", SIX_VIEWS),
        0.0,
    )
    assert math.isclose(
        action_compatibility("orbit_left", "V0", "V1", SIX_VIEWS),
        0.0,
    )


def test_adjacent_direction_retains_bounded_compatibility() -> None:
    assert math.isclose(
        action_compatibility("orbit_right_down", "V0", "V1", SIX_VIEWS),
        0.25,
    )
