import numpy as np
import pytest

from space_env.core.relative import cw_stm
from space_env.industrial.safety import (
    check_cw_interval,
    cw_speed_bound,
    interval_separation,
)


def test_crossing_between_safe_endpoints_is_detected():
    result = interval_separation(lambda t: [100 - 20 * t, 0, 0], 10.0, 10.0, 20.0)
    assert result["status"] == "unsafe"
    assert result["sampled_min_m"] == 0


def test_budget_exhaustion_never_means_safe():
    result = interval_separation(
        lambda t: [100 - 20 * t, 0, 0], 10.0, 10.0, 20.0, max_evaluations=2
    )
    assert result["status"] == "indeterminate"


def test_safe_interval_with_known_analytic_minimum():
    result = interval_separation(lambda t: [100 - 20 * t, 30, 0], 10.0, 20.0, 20.0)
    assert result["status"] == "safe"
    assert 20 <= result["lower_bound_m"] <= 30


def test_tangent_is_not_falsely_declared_safe():
    result = interval_separation(
        lambda t: [100 - 20 * t, 10, 0], 10.0, 10.0, 20.0, max_evaluations=33
    )
    assert result["status"] == "indeterminate"


@pytest.mark.parametrize("seed", range(8))
def test_cw_velocity_bound_against_independent_ode(seed):
    from scipy.integrate import solve_ivp

    state = np.random.default_rng(seed).normal(size=6) * [100, 100, 100, 1, 1, 1]
    n, duration = 0.0011, 120.0

    def rhs(t, x):
        return [*x[3:], 3 * n * n * x[0] + 2 * n * x[4], -2 * n * x[3], -n * n * x[2]]

    solution = solve_ivp(
        rhs,
        (0, duration),
        state,
        t_eval=np.linspace(0, duration, 121),
        rtol=1e-11,
        atol=1e-11,
    )
    assert np.max(np.linalg.norm(solution.y[3:], axis=0)) <= cw_speed_bound(
        state, n, duration
    )
    np.testing.assert_allclose(
        solution.y[:, -1], cw_stm(n, duration) @ state, atol=1e-7
    )


@pytest.mark.parametrize("bad", [True, -1.0, float("nan"), float("inf")])
def test_invalid_bounds_rejected(bad):
    with pytest.raises(ValueError):
        interval_separation(lambda t: [100, 0, 0], 1.0, 10.0, bad)


def test_cw_crossing_not_missed():
    assert (
        check_cw_interval([100, 0, 0, -20, 0, 0], 0.0, 10.0, 10.0)["status"] == "unsafe"
    )
