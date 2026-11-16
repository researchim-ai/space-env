"""Catalogue identity, runtime compatibility and native execution checks."""

import pytest

from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSession
from space_env.tasks.catalog import mission_catalog, mission_recipe, run_mission


def test_catalog_identity_and_schedule():
    catalog = mission_catalog()
    assert len(catalog["missions"]) == 28
    assert len({r["id"] for r in catalog["missions"]}) == 28
    for row in catalog["missions"]:
        recipe = mission_recipe(row["id"])
        assert row["recipe_sha256"] == digest(recipe)
        end = 0.0
        actions = {a["id"] for a in recipe["task"]["actions"]} | {"wait"}
        for phase in row["schedule"]:
            assert phase["start_sec"] == end
            assert phase["end_sec"] > end
            assert set(phase["action_ids"]) <= actions
            end = phase["end_sec"]


@pytest.mark.parametrize("mission_id", [r["id"] for r in mission_catalog()["missions"]])
def test_all_recipes_construct(mission_id):
    recipe = mission_recipe(mission_id)
    session = TaskSession(recipe["scenario"], recipe["task"])
    assert session._runtime.time_sec == 0.0


def test_defensive_exports():
    recipe = mission_recipe("finite/finite-central")
    recipe["task"]["actions"].clear()
    assert mission_recipe("finite/finite-central")["task"]["actions"]
    with pytest.raises(ValueError, match="Unknown mission"):
        mission_recipe("docking")
    with pytest.raises(ValueError, match="policy"):
        run_mission("finite/finite-central", policy="invented")


@pytest.mark.parametrize("policy", ["wait", "reference"])
def test_native_execution(policy):
    record = run_mission("finite/finite-central", policy=policy)
    assert record["metrics"]["success"] == (policy == "reference")
