"""Versioned train-only regret classification with explicit convergence evidence.

This selector learns from native payoffs of the three frozen strategy choices.
It is not a command-level RL policy or an expert-teacher qualification.
"""

from __future__ import annotations

import numpy as np

from space_env.planning.network_benchmark import (
    FEATURES,
    STRATEGIES,
    _training_identity,
    training_arrays,
)
from space_env.simulation.snapshot import digest
from space_env.validation.network_evidence import seal


def fit_regret_selector(suite, train_views, rows, *, seed=0):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid training seed")
    spec, train, raw, payoffs = training_arrays(suite, train_views, rows)
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    center[0], scale[0] = 0.0, 1.0
    scale[scale < 1e-12] = 1.0
    x = np.clip((raw - center) / scale, -8.0, 8.0)
    target = np.isclose(
        payoffs, payoffs.max(axis=1, keepdims=True), atol=1e-12, rtol=0
    ).astype(float)
    target /= target.sum(axis=1, keepdims=True)
    weights = np.random.default_rng(seed).normal(
        0.0, 1e-4, (len(FEATURES), len(STRATEGIES))
    )
    learning_rate, regularization, tolerance, maximum = 0.2, 0.01, 1e-6, 2000

    def objective(w):
        logits = x @ w
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        probabilities = np.exp(log_probabilities)
        loss = (
            -float(np.mean(np.sum(target * log_probabilities, axis=1)))
            + regularization * float(np.sum(w * w)) / 2
        )
        gradient = x.T @ (probabilities - target) / len(train) + regularization * w
        return loss, gradient, logits

    initial_loss = objective(weights)[0]
    iterations = 0
    for _ in range(maximum):
        _, gradient, _ = objective(weights)
        if float(np.linalg.norm(gradient)) <= tolerance:
            break
        weights -= learning_rate * gradient
        iterations += 1
    loss, gradient, logits = objective(weights)
    greedy = payoffs[np.arange(len(train)), np.argmax(logits, axis=1)]
    fixed = payoffs.mean(axis=0)
    best_available = float(payoffs.max(axis=1).mean())
    norm = float(np.linalg.norm(gradient))
    diagnostics = {
        "initial_loss": initial_loss,
        "final_loss": loss,
        "gradient_norm": norm,
        "gradient_tolerance": tolerance,
        "numerically_converged": norm <= tolerance,
        "mean_train_reward": float(greedy.mean()),
        "best_fixed_train_strategy": STRATEGIES[int(np.argmax(fixed))],
        "best_fixed_train_reward": float(fixed.max()),
        "strategy_class_train_upper_bound": best_available,
        "train_regret_to_strategy_class": max(
            0.0, best_available - float(greedy.mean())
        ),
        "scope": "training optimization and three-strategy class only; not held-out/native mission optimality",
    }
    return seal(
        {
            "schema": "space-env.network-selector/2",
            "training_method": "regret_softmax",
            "seed": seed,
            "features": list(FEATURES),
            "strategies": list(STRATEGIES),
            "weights": weights.tolist(),
            "transform": {
                "schema": "space-env.train-standardization/1",
                "center": center.tolist(),
                "scale": scale.tolist(),
                "clip": 8.0,
            },
            "training_input_sha256": _training_identity(spec),
            "training_cases": [c.id for c in train],
            "training_rows_sha256": digest(
                sorted(rows, key=lambda e: (e["case"], e["strategy"]))
            ),
            "iterations": iterations,
            "maximum_iterations": maximum,
            "learning_rate": learning_rate,
            "l2": regularization,
            "diagnostics": diagnostics,
            "scope": "train-only regularized classification of native best-strategy payoffs, fixed optimizer settings; no command-level RL, test tuning or expert qualification",
        }
    )
