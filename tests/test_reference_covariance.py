"""Variational dynamics, continuous noise and maneuver covariance contracts."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from space_env.reference import (
    EOPSeries,
    Epoch,
    ForceModel,
    FrameContext,
    ReferenceState,
    propagate,
)
from space_env.reference.access import ReferenceEphemeris
from space_env.reference.covariance import (
    CovarianceArc,
    NavigationSettings,
    state_jacobian,
)
from space_env.runtime import (
    AccessPolicy,
    MissionRuntime,
    RuntimeContractError,
    dynamic_reference_scenario,
)
from tests import test_reference_dynamic as dynamic_fixtures


@pytest.fixture(scope="module")
def nominal():
    initial = ReferenceState(
        Epoch.utc("2026-07-18T12:00:00Z"), [7e6, 0, 0], [0, 7400, 1400]
    )
    context = FrameContext(EOPSeries.packaged())
    result = propagate(initial, 60, context, sample_sec=1.0, max_step_sec=5.0)
    return initial, context, ReferenceEphemeris.from_result(result)


def settings(*, noise=0.0, impulse_noise=0.0):
    return NavigationSettings(
        initial_covariance_si=np.diag([100.0, 200.0, 300.0, 0.01, 0.02, 0.03]).tolist(),
        acceleration_psd_m2_s3=(np.eye(3) * noise).tolist(),
        impulse_velocity_cov_m2_s2=(np.eye(3) * impulse_noise).tolist(),
    )


@pytest.fixture(scope="module")
def dynamic_spec():
    result = copy.deepcopy(dynamic_fixtures.dynamic_spec.__wrapped__())
    for orbit in result["orbits"].values():
        orbit["navigation"] = settings(noise=1e-7, impulse_noise=0.04).model_dump(
            mode="json"
        )
    return result


def model(spec):
    return MissionRuntime(
        dynamic_reference_scenario(spec, id="navigation", lineage="B02-Q1"),
        access=AccessPolicy(truth=True, forecast=True),
    )


def test_stm_matches_six_independently_perturbed_orbits(nominal):
    initial, context, ephemeris = nominal
    arc = CovarianceArc(ephemeris, settings())
    finite_difference = np.empty((6, 6))
    state = np.concatenate((initial.r_m, initial.v_m_s))
    for column in range(6):
        delta = np.zeros(6)
        step = 1.0 if column < 3 else 0.001
        delta[column] = step
        finals = []
        for sign in (1, -1):
            shifted = state + sign * delta
            record = propagate(
                ReferenceState(initial.epoch, shifted[:3], shifted[3:]),
                60,
                context,
                sample_sec=1.0,
                max_step_sec=5.0,
            )
            finals.append(np.array(record["states_si"][-1]))
        finite_difference[:, column] = (finals[0] - finals[1]) / (2 * step)
    np.testing.assert_allclose(
        arc.transition(60), finite_difference, atol=5e-6, rtol=1e-6
    )
    np.testing.assert_array_equal(arc.stm(0), np.eye(6))


def test_deterministic_transport_and_noise_are_separate_and_psd(nominal):
    _, _, ephemeris = nominal
    deterministic, noisy = (
        CovarianceArc(ephemeris, settings()),
        CovarianceArc(ephemeris, settings(noise=1e-5)),
    )
    for clock in (0.0, 0.5, 10.0, 60.0):
        phi = deterministic.transition(clock)
        np.testing.assert_allclose(
            deterministic.covariance(clock),
            phi @ deterministic.initial_covariance @ phi.T,
            atol=1e-9,
        )
        increment = noisy.covariance(clock) - deterministic.covariance(clock)
        assert np.linalg.eigvalsh(increment).min() >= -1e-9
        assert np.linalg.eigvalsh(noisy.covariance(clock)).min() >= 0
    assert np.trace(noisy.covariance(60)) > np.trace(deterministic.covariance(60))


def test_white_acceleration_diffusion_matches_linear_analytic_solution(
    nominal, monkeypatch
):
    # A constant-velocity variational model isolates the Q discretization from
    # orbit forces: Wrr=Q*t^3/3, Wrv=Q*t^2/2, Wvv=Q*t.
    from space_env.reference import covariance as module

    _, _, ephemeris = nominal
    linear = np.zeros((6, 6))
    linear[:3, 3:] = np.eye(3)
    monkeypatch.setattr(module, "state_jacobian", lambda *a, **kw: linear)
    q = 1e-4 * np.eye(3)
    arc = CovarianceArc(ephemeris, settings(noise=1e-4))
    time = 13.25
    expected = np.block(
        [[q * time**3 / 3, q * time**2 / 2], [q * time**2 / 2, q * time]]
    )
    np.testing.assert_allclose(arc.diffusion(time), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "kind", ["nonsymmetric", "negative", "shape", "nan", "bool", "frame", "extra"]
)
def test_navigation_input_rejects_invalid_contracts(kind):
    value = settings().model_dump(mode="json")
    if kind == "nonsymmetric":
        value["initial_covariance_si"][0][1] = 1.0
    elif kind == "negative":
        value["acceleration_psd_m2_s3"][0][0] = -1.0
    elif kind == "shape":
        value["initial_covariance_si"] = [[1.0]]
    elif kind == "nan":
        value["impulse_velocity_cov_m2_s2"][0][0] = float("nan")
    elif kind == "bool":
        value["initial_covariance_si"][0][0] = True
    elif kind == "frame":
        value["frame"] = "ITRS"
    else:
        value["mass_sigma_kg"] = 1.0
    with pytest.raises(ValueError):
        NavigationSettings.model_validate(value)


def test_impulse_noise_jump_and_global_stm_are_not_reset(dynamic_spec):
    runtime = model(dynamic_spec)
    runtime.advance(1, expected_revision=0, idempotency_key="coast")
    provider = runtime._backend.mission.ephemerides["a"]
    covariance, stm, past = (
        provider.covariance(1),
        provider.stm(1),
        provider.covariance(0.5),
    )
    runtime.submit(
        [dynamic_fixtures.impulse()], expected_revision=1, idempotency_key="kick"
    )
    expected = covariance.copy()
    expected[3:, 3:] += 0.04 * np.eye(3)
    np.testing.assert_allclose(provider.covariance(1), expected, atol=1e-12)
    np.testing.assert_allclose(provider.stm(1), stm, atol=1e-12)
    np.testing.assert_array_equal(provider.covariance(0.5), past)
    truth = runtime.observe(view="truth")["backend"]["states"]["a"]
    np.testing.assert_allclose(truth["covariance_si"], expected)


def test_mid_burn_checkpoint_cancel_and_frame_covariance_roundtrip(
    dynamic_spec, tmp_path
):
    runtime = model(dynamic_spec)
    runtime.submit(
        [dynamic_fixtures.burn()], expected_revision=0, idempotency_key="burn"
    )
    runtime.advance(2, expected_revision=1, idempotency_key="half")
    provider = runtime._backend.mission.ephemerides["a"]
    before, stm = provider.covariance(2), provider.stm(2)
    path = tmp_path / "navigation.json"
    runtime.save(path)
    restored = MissionRuntime.load(path, access=AccessPolicy(truth=True))
    for item in (runtime, restored):
        item.cancel("burn", expected_revision=2, idempotency_key="cancel")
        p = item._backend.mission.ephemerides["a"]
        np.testing.assert_allclose(p.covariance(2), before, atol=1e-9)
        np.testing.assert_allclose(p.stm(2), stm, atol=1e-9)
        item.advance(2, expected_revision=3, idempotency_key="coast")
    assert restored.checkpoint() == runtime.checkpoint()
    epoch = provider.epoch.shifted(4)
    inertial = provider.state(4)
    fixed = provider.context.transform(
        inertial,
        epoch,
        source="GCRS",
        destination="ITRS",
        covariance=provider.covariance(4),
    )
    inverse = provider.context.transform(
        fixed["state_si"],
        epoch,
        source="ITRS",
        destination="GCRS",
        covariance=fixed["covariance_si"],
    )
    np.testing.assert_allclose(
        inverse["covariance_si"], provider.covariance(4), atol=1e-8
    )


def test_navigation_capabilities_are_explicit_per_actor(dynamic_spec):
    runtime = model(dynamic_spec)
    assert (
        "state.covariance_propagation" in runtime.describe_capabilities()["supported"]
    )
    data = copy.deepcopy(dynamic_spec)
    del data["orbits"]["b"]["navigation"]
    partial = model(data)
    assert (
        "state.covariance_propagation"
        not in partial.describe_capabilities()["supported"]
    )
    assert (
        "state.covariance.partial_actors"
        in partial.describe_capabilities()["supported"]
    )
    assert "covariance_si" not in partial.observe("b")["actors"]["b"]["orbit"]
    with pytest.raises(ValueError, match="capability"):
        partial._backend.mission.ephemerides["b"].covariance(0)
    scenario = dynamic_reference_scenario(data, id="required", lineage="B02-Q1")
    scenario["required_capabilities"] = ["state.covariance_propagation"]
    with pytest.raises(RuntimeContractError, match="capabilities"):
        MissionRuntime(scenario)


def test_numerical_partials_include_rotating_drag_and_all_force_switches(nominal):
    initial, context, _ = nominal
    state = np.concatenate((initial.r_m, initial.v_m_s))
    forces = ForceModel(j2=True, drag=True, srp=True, third_body=True)
    fine = state_jacobian(
        state,
        initial.epoch,
        context,
        forces,
        position_step_m=0.5,
        velocity_step_m_s=0.0005,
    )
    coarse = state_jacobian(
        state,
        initial.epoch,
        context,
        forces,
        position_step_m=2.0,
        velocity_step_m_s=0.002,
    )
    np.testing.assert_allclose(fine, coarse, rtol=1e-5, atol=2e-11)
    assert np.linalg.norm(fine[3:, 3:]) > 0


def test_covariance_query_bounds_and_copy_isolation(nominal):
    _, _, ephemeris = nominal
    arc = CovarianceArc(ephemeris, settings())
    for clock in (True, -1.0, 61.0, float("nan")):
        with pytest.raises(ValueError):
            arc.covariance(clock)
    value = arc.covariance(5)
    value[:] = 0
    assert np.trace(arc.covariance(5)) > 0


def test_independent_sigma_point_covariance_report():
    from space_env.validation.covariance import covariance_prediction_report

    report = covariance_prediction_report()
    assert report["passed"], report
    assert "not random Monte Carlo" in report["ensemble"]


def test_failed_maneuver_batch_rolls_back_covariance_and_global_stm(
    dynamic_spec, monkeypatch
):
    runtime = model(dynamic_spec)
    original = runtime._backend.admit

    def fail_second(command):
        if command.actor_id == "b":
            raise RuntimeError("actuator")
        return original(command)

    monkeypatch.setattr(runtime._backend, "admit", fail_second)
    before = runtime.checkpoint()
    covariance = runtime._backend.mission.ephemerides["a"].covariance(10)
    with pytest.raises(RuntimeError, match="actuator"):
        runtime.submit(
            [dynamic_fixtures.impulse(), dynamic_fixtures.impulse(id="b", actor="b")],
            expected_revision=0,
            idempotency_key="batch",
        )
    assert runtime.checkpoint() == before
    np.testing.assert_array_equal(
        runtime._backend.mission.ephemerides["a"].covariance(10), covariance
    )


def test_burn_end_global_stm_and_covariance_composition(dynamic_spec):
    runtime = model(dynamic_spec)
    runtime.submit(
        [dynamic_fixtures.burn()], expected_revision=0, idempotency_key="burn"
    )
    provider = runtime._backend.mission.ephemerides["a"]
    burn_arc, coast_arc = [arc for _, arc in provider.navigation_arcs]
    np.testing.assert_allclose(coast_arc.stm(0), burn_arc.stm(4), atol=1e-12)
    np.testing.assert_allclose(
        coast_arc.covariance(0), burn_arc.covariance(4), atol=1e-9
    )
    np.testing.assert_allclose(
        provider.stm(5), coast_arc.transition(1) @ burn_arc.stm(4), atol=1e-10
    )


def test_central_stm_preserves_symplectic_structure_and_singular_zero_covariance(
    nominal,
):
    _, _, ephemeris = nominal
    zero = NavigationSettings(initial_covariance_si=np.zeros((6, 6)).tolist())
    arc = CovarianceArc(ephemeris, zero)
    symplectic = np.block(
        [[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]]
    )
    phi = arc.stm(60)
    np.testing.assert_allclose(phi.T @ symplectic @ phi, symplectic, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(arc.covariance(13.5), np.zeros((6, 6)))


def test_perturbed_covariance_converges_with_linearization_steps(nominal):
    initial, context, _ = nominal
    forces = ForceModel(j2=True, drag=True, srp=True, third_body=True)
    ephemeris = ReferenceEphemeris.from_result(
        propagate(initial, 10, context, model=forces, sample_sec=1.0)
    )
    fine = CovarianceArc(
        ephemeris,
        settings().model_copy(
            update={
                "position_difference_step_m": 0.5,
                "velocity_difference_step_m_s": 0.0005,
                "max_step_sec": 0.5,
            }
        ),
        model=forces,
    )
    coarse = CovarianceArc(
        ephemeris,
        settings().model_copy(
            update={
                "position_difference_step_m": 2.0,
                "velocity_difference_step_m_s": 0.002,
                "max_step_sec": 1.0,
            }
        ),
        model=forces,
    )
    np.testing.assert_allclose(
        fine.covariance(10), coarse.covariance(10), rtol=1e-6, atol=1e-7
    )


def test_positive_diagonal_does_not_hide_invalid_cross_covariance():
    value = settings().model_dump(mode="json")
    value["initial_covariance_si"][0][1] = value["initial_covariance_si"][1][0] = 1000.0
    with pytest.raises(ValueError, match="semidefinite"):
        NavigationSettings.model_validate(value)


def test_covariance_rejects_unsealed_and_inside_earth_inputs(nominal):
    initial, context, ephemeris = nominal
    with pytest.raises(ValueError, match="sealed"):
        CovarianceArc(object(), settings())
    with pytest.raises(ValueError, match="Earth"):
        state_jacobian(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], initial.epoch, context, ForceModel()
        )
    with pytest.raises(ValueError, match="ForceModel"):
        CovarianceArc(ephemeris, settings(), model={})


def test_covariance_force_and_thrust_models_must_match_the_sealed_nominal_arc(nominal):
    initial, context, ephemeris = nominal
    with pytest.raises(ValueError, match="force model differs"):
        CovarianceArc(ephemeris, settings(), model=ForceModel(j2=True))
    from space_env.reference import FiniteThrust

    thrust = FiniteThrust(2.0, 300.0, 90.0, (0.0, 1.0, 0.0))
    powered = ReferenceEphemeris.from_result(
        propagate(initial, 4, context, sample_sec=1.0, thrust=thrust)
    )
    with pytest.raises(ValueError, match="thrust model differs"):
        CovarianceArc(powered, settings())
    with pytest.raises(ValueError, match="thrust model differs"):
        CovarianceArc(ephemeris, settings(), thrust=thrust)
    assert np.isfinite(
        CovarianceArc(powered, settings(), thrust=thrust).covariance(4)
    ).all()


def test_covariance_solver_evaluation_budget_is_enforced(nominal):
    _, _, ephemeris = nominal
    limited = settings().model_copy(
        update={"max_evaluations": 100, "max_step_sec": 0.1}
    )
    with pytest.raises(ValueError, match="evaluation budget"):
        CovarianceArc(ephemeris, limited)
