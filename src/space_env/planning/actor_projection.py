"""Shared online/offline actor observation contract; native binding retained."""

import copy

from space_env.simulation.snapshot import digest
from space_env.tasks import TaskSpec


def actor_roles(task):
    spec = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
    return tuple(sorted(k for k, v in spec.roles.items() if v.actors))


def actor_catalogues(task, roles):
    spec = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
    result = {}
    for role in roles:
        if role not in spec.roles or not spec.roles[role].actors:
            raise ValueError("actor role required")
        actions = [a for a in spec.actions if a.role == role]
        if any(a.kind == "inject" for a in actions):
            raise ValueError(
                "operator injection cannot be exposed as actor policy action"
            )
        result[role] = ("wait", *(a.id for a in actions))
    return result


def project_actor_observation(value, task, role, *, catalogue_version=None):
    spec = task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task)
    actor_catalogues(spec, (role,))
    if (
        value.get("schema") != "space-env.task-observation/1"
        or value.get("role") != role
        or value.get("binding")
        != digest({k: v for k, v in value.items() if k != "binding"})
    ):
        raise ValueError("invalid native actor observation/binding")
    obs = copy.deepcopy(value)
    if len(obs["catalogue"]) != len(obs["action_mask"]):
        raise ValueError("native role catalogue/mask mismatch")
    owners = {a.id: a.role for a in spec.actions}
    indices = [
        i
        for i, a in enumerate(obs["catalogue"])
        if a["id"] == "wait" or owners.get(a["id"]) == role
    ]
    obs["catalogue"] = [obs["catalogue"][i] for i in indices]
    obs["action_mask"] = [obs["action_mask"][i] for i in indices]
    obs["native_observation_binding"] = obs.pop("binding")
    obs["schema"] = "space-env.incident-actor-observation/1"
    if catalogue_version is not None:
        if (
            not isinstance(catalogue_version, str)
            or len(catalogue_version) != 64
            or any(c not in "0123456789abcdef" for c in catalogue_version)
        ):
            raise ValueError("invalid actor catalogue version")
        obs["schema"] = "space-env.versioned-actor-observation/1"
        obs["catalogue_version"] = catalogue_version
    obs["binding"] = digest(obs)
    return obs
