"""Успех эвристики сохраняется при росте числа аппаратов."""
from space_env.eval.scale import scale_report


def test_expert_succeeds_as_the_fleet_grows():
    rows = scale_report()
    assert {(row["task"], row["count"]) for row in rows} == {
        ("station", 2),
        ("station", 4),
        ("station", 6),
        ("orders", 2),
        ("orders", 3),
        ("orders", 4),
        ("shift", 4),
        ("shift", 6),
    }
    assert all(row["success"] for row in rows)
