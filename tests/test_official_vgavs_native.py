from scripts.evaluate_official_vgavs_native import parse_native_action


def test_uses_last_complete_action_not_initial_guess():
    text = "<head>0</head><fwd>0</fwd><view>0</view><think>refine</think><head>-90</head><fwd>50</fwd><view>90</view>"
    assert parse_native_action(text) == {
        "heading_degrees": -90, "forward_centimeters": 50, "view_degrees": 90,
    }


def test_invalid_or_truncated_action_is_not_silently_stay():
    assert parse_native_action("<head>90</head><fwd>50</fwd>") is None
    assert parse_native_action("<head>200</head><fwd>50</fwd><view>0</view>") is None
    assert parse_native_action("<head>0</head><fwd>-3</fwd><view>0</view>") is None
    assert parse_native_action(
        "<head>0</head><fwd>0</fwd><view>0</view><think>refine</think><head>90</head>"
    ) is None
