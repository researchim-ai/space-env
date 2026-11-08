"""Synthetic optimizer checks, explicitly distinct from native mission evidence."""

import copy

import numpy as np
import pytest

from space_env.planning import network_benchmark as benchmark
from space_env.planning.network_selector import fit_regret_selector
from space_env.planning.network_selector_study import selector_suite
from space_env.validation.network_evidence import seal
from tests.test_network_benchmark import training_data


def test_train_only_normalization_convergence_and_regret():
    suite, views, rows = training_data()
    model = fit_regret_selector(suite, views, rows)
    raw = np.stack([benchmark.features(v) for v in views.values()])
    expected = raw.mean(axis=0)
    expected[0] = 0
    np.testing.assert_allclose(model["transform"]["center"], expected)
    diagnostics = model["diagnostics"]
    assert diagnostics["final_loss"] < diagnostics["initial_loss"]
    assert diagnostics["train_regret_to_strategy_class"] == 0
    assert diagnostics["numerically_converged"] == (
        diagnostics["gradient_norm"] <= diagnostics["gradient_tolerance"]
    )
    assert model == fit_regret_selector(suite, views, rows)
    assert all(
        benchmark.choose_strategy(model, v) in benchmark.STRATEGIES
        for v in views.values()
    )


def test_rejects_test_rows_missing_matrix_and_invalid_transforms():
    suite, views, rows = training_data()
    with pytest.raises(ValueError):
        fit_regret_selector(suite, views, rows[:-1])
    bad = copy.deepcopy(rows)
    bad[0]["case"] = suite["cases"][-1]["id"]
    with pytest.raises(ValueError):
        fit_regret_selector(suite, views, bad)
    model = fit_regret_selector(suite, views, rows)
    for value in (0, -1, float("inf")):
        bad = copy.deepcopy(model)
        bad["transform"]["scale"][1] = value
        # infinity cannot be serialized as a valid sealed artifact either.
        with pytest.raises(ValueError):
            benchmark.choose_strategy(
                seal({k: v for k, v in bad.items() if k != "sha256"}),
                next(iter(views.values())),
            )


def test_new_holdout_is_frozen_and_preserves_historical_training():
    from space_env.planning.network_demand_benchmark import demand_suite

    old, new = demand_suite(), selector_suite()
    assert new == selector_suite()
    assert [c for c in old["cases"] if c["split"] != "test"] == [
        c for c in new["cases"] if c["split"] != "test"
    ]
    assert not {c["geometry_sha256"] for c in old["cases"]} & {
        c["geometry_sha256"] for c in new["cases"] if c["split"] == "test"
    }
