"""Команды миссии. Тот же исполнитель, что у политики и среды."""
from __future__ import annotations

import argparse
import json

from space_env.mission.session import mission_report


def main(argv: list[str] | None = None) -> None:
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="space_env")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("scenario")
    run.add_argument("--policy", default="priority", choices=("fifo", "deadline", "priority"))
    run.add_argument("--step", type=float, default=60.0)
    validate = sub.add_parser("validate")
    validate.add_argument("scenario")
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("suite")
    replay_command = sub.add_parser("replay")
    replay_command.add_argument("record")
    portable = sub.add_parser("record-commands", help="Record a portable direct-command experiment")
    portable.add_argument("scenario")
    portable.add_argument("plan", help="JSON object with steps: [{dt_sec, commands: [...]}]")
    portable.add_argument("output")
    portable.add_argument("--lineage", required=True)
    verify_run = sub.add_parser("replay-run", help="Verify a portable command experiment")
    verify_run.add_argument("record")
    transitions = sub.add_parser("export-transitions", help="Export replay-verified behavior transitions")
    transitions.add_argument("record")
    transitions.add_argument("output")
    agent = sub.add_parser("record-tools", help="Record a portable mission-tool script")
    agent.add_argument("scenario")
    agent.add_argument("script", help="JSON with task and calls [{name, arguments, optional call_id}]")
    agent.add_argument("output")
    agent.add_argument("--lineage", required=True)
    agent.add_argument("--seed", type=int, default=0)
    agent.add_argument("--call-budget", type=int, default=0)
    verify_agent = sub.add_parser("replay-tools", help="Verify a portable tool dialogue")
    verify_agent.add_argument("record")
    dialogue = sub.add_parser("export-dialogue", help="Export one replay-verified tool conversation")
    dialogue.add_argument("record")
    dialogue.add_argument("output")
    teacher = sub.add_parser("record-policy", help="Run a bounded heuristic through recorded mission tools")
    teacher.add_argument("scenario")
    teacher.add_argument("output")
    teacher.add_argument("--lineage", required=True)
    teacher.add_argument("--policy", default="priority", choices=("fifo", "deadline", "priority", "rolling"))
    teacher.add_argument("--step", type=float, default=30.)
    teacher.add_argument("--max-steps", type=int, default=100)
    teacher.add_argument("--seed", type=int, default=0)
    teacher.add_argument("--call-budget", type=int, default=0)
    physics = sub.add_parser("physics-reference", help="Measured GCRS frame and orbit reference checks")
    physics.add_argument("--long", action="store_true", help="Include circular two-body 24/72h arcs")
    physics.add_argument("--crosscheck", action="store_true", help="Compare force formulas and J2 arcs with Boinor/LSODA")
    reference = sub.add_parser("reference-propagate", help="Propagate an explicit GCRS state with DOP853")
    reference.add_argument("spec")
    reference.add_argument("output")
    reference.add_argument("--allow-predictions", action="store_true")
    access = sub.add_parser("reference-access", help="Compute GCRS geometric station windows from a sealed orbit")
    access.add_argument("orbit")
    access.add_argument("stations", help="JSON with a stations list of explicit GroundStation fields")
    access.add_argument("output")
    access.add_argument("--scan", type=float, default=30.)
    ref_run = sub.add_parser("reference-run", help="Execute fixed-ephemeris downlink resource operations")
    ref_run.add_argument("spec")
    ref_run.add_argument("plan")
    ref_run.add_argument("output")
    ref_run.add_argument("--lineage", required=True)
    ref_replay = sub.add_parser("reference-replay", help="Replay a portable downlink operations run")
    ref_replay.add_argument("record")
    ref_plan = sub.add_parser("reference-plan", help="Generate and execute-verify a greedy downlink schedule")
    ref_plan.add_argument("spec")
    ref_plan.add_argument("output", help="Executable plan JSON for reference-run")
    ref_plan.add_argument("--report", required=True, help="Schedule verification and unmet-task report")
    ref_plan.add_argument("--decision-sec", type=float, default=5.)
    ref_plan.add_argument("--max-steps", type=int, default=10000)
    ref_plan.add_argument("--select-products", action="store_true")
    ref_replan = sub.add_parser("reference-replan", help="Plan remaining work from a verified portable run")
    ref_replan.add_argument("record")
    ref_replan.add_argument("output")
    ref_replan.add_argument("--report", required=True)
    ref_replan.add_argument("--decision-sec", type=float, default=5.)
    ref_replan.add_argument("--max-steps", type=int, default=10000)
    ref_resume = sub.add_parser("reference-resume", help="Append a continuation plan to a verified portable run")
    ref_resume.add_argument("record")
    ref_resume.add_argument("plan")
    ref_resume.add_argument("output")
    for command in ("reference-events", "reference-adapt"):
        event_command = sub.add_parser(command, help="Inject causal events into a verified reference run")
        event_command.add_argument("record")
        event_command.add_argument("events", help="JSON object with an events list")
        event_command.add_argument("output")
        if command == "reference-adapt":
            event_command.add_argument("--report", required=True)
            event_command.add_argument("--decision-sec", type=float, default=5.)
            event_command.add_argument("--max-steps", type=int, default=10000)
    oracle = sub.add_parser("reference-optimal", help="Bounded exact nonpreemptive calendar optimum and matching-class regret")
    oracle.add_argument("spec")
    oracle.add_argument("output", help="Optimal executable plan")
    oracle.add_argument("--report", required=True)
    oracle.add_argument("--max-states", type=int, default=65536)
    calendar = sub.add_parser("reference-benchmark", help="Run a self-contained bounded calendar benchmark")
    calendar.add_argument("suite")
    calendar.add_argument("output")
    calendar.add_argument("--max-states", type=int, default=65536)
    calendar.add_argument("--total-states", type=int, default=262144)
    calendar_verify = sub.add_parser("reference-verify-benchmark", help="Recompute and verify a portable calendar benchmark")
    calendar_verify.add_argument("record")
    qualification = sub.add_parser("reference-qualify-orbits", help="compare frozen external J2 arcs and finite-thrust equations")
    qualification.add_argument("output")
    for name in ("mission-validate", "mission-run", "mission-resume", "mission-replay", "mission-schema"):
        unified = sub.add_parser(name, help="Scenario v3 unified runtime workflow")
        if name in {"mission-validate", "mission-run"}:
            unified.add_argument("scenario")
        if name in {"mission-resume", "mission-replay"}:
            unified.add_argument("record")
        if name in {"mission-run", "mission-resume"}:
            unified.add_argument("plan")
        if name in {"mission-schema", "mission-run", "mission-resume"}:
            unified.add_argument("output")
    for name in ("task-run","task-resume","task-replay","task-schema","task-export"):
        task_parser=sub.add_parser(name,help="Verified runtime-backed task workflow")
        if name=="task-run":
            task_parser.add_argument("scenario")
            task_parser.add_argument("task")
        if name in {"task-resume","task-replay","task-export"}:task_parser.add_argument("record")
        if name in {"task-run","task-resume"}:task_parser.add_argument("plan")
        if name!="task-replay":task_parser.add_argument("output")
        if name=="task-export":
            task_parser.add_argument("--format",choices=["rl","sft"],default="rl")
            task_parser.add_argument("--role",default="@central")
    for name in ("planning-run", "planning-resume", "planning-replay", "planning-evaluate", "planning-verify-evaluation"):
        planning_parser = sub.add_parser(name, help="Causal baseline planning and evaluation")
        if name == "planning-run":
            for argument in ("scenario", "task", "policy"):
                planning_parser.add_argument(argument)
            planning_parser.add_argument("--seed", type=int, default=None)
        elif name == "planning-evaluate":
            planning_parser.add_argument("manifest")
            planning_parser.add_argument("--artifacts", default=None)
        else:
            planning_parser.add_argument("record")
        if name not in {"planning-replay", "planning-verify-evaluation"}:
            planning_parser.add_argument("output")
        if name in {"planning-run", "planning-resume"}:
            planning_parser.add_argument("--decisions", type=int, default=None)
    for name in ("factory-schema", "factory-compile", "factory-build", "factory-verify", "factory-register"):
        factory_parser = sub.add_parser(name, help="Versioned synthetic mission/data factory")
        if name in {"factory-verify", "factory-register"}:
            factory_parser.add_argument("bundle")
            if name == "factory-verify": factory_parser.add_argument("--integrity-only", action="store_true")
            else: factory_parser.add_argument("registry")
        else:
            if name != "factory-schema": factory_parser.add_argument("spec")
            factory_parser.add_argument("output")
    scale_parser = sub.add_parser("scale-benchmark", help="Frozen isolated-process scale measurements")
    scale_parser.add_argument("manifest")
    scale_parser.add_argument("output")
    args = parser.parse_args(argv)
    if args.command == "scale-benchmark":
        from space_env.scaling.benchmark import run_scale
        try:
            result = run_scale(json.loads(Path(args.manifest).read_text()), args.output)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            raise SystemExit(1) from exc
        print(json.dumps(result, ensure_ascii=False))
        if not result["ok"]:
            raise SystemExit(1)
        return
    if args.command.startswith("factory-"):
        from space_env.factory.cli import run_command as factory_command

        try:
            result = factory_command(args)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": {"code": "factory_contract_error", "message": str(exc)}}))
            raise SystemExit(1) from exc
        print(json.dumps(result, ensure_ascii=False))
        if not result.get("ok", True): raise SystemExit(1)
        return
    if args.command.startswith("planning-"):
        from space_env.planning.cli import run_command as planning_command

        try:
            result = planning_command(args)
        except ValueError as exc:
            print(json.dumps({"ok": False, "error": {"code": "planning_contract_error", "message": str(exc)}}))
            raise SystemExit(1) from exc
        print(json.dumps(result, ensure_ascii=False))
        if not result.get("ok", True):
            raise SystemExit(1)
        return
    if args.command.startswith("task-"):
        from space_env.tasks.cli import run_command as task_command
        try:
            result=task_command(args)
        except ValueError as exc:
            print(json.dumps({"ok":False,"error":{"code":"task_contract_error","message":str(exc)}}))
            raise SystemExit(1) from exc
        print(json.dumps(result,ensure_ascii=False))
        return

    if args.command == "reference-qualify-orbits":
        from space_env.simulation.snapshot import atomic_write, canonical_json
        from space_env.validation.covariance import covariance_prediction_report
        from space_env.validation.geometry_events import geometry_event_report
        from space_env.validation.maneuvers import finite_burn_report
        from space_env.validation.orbit_fixtures import verify_packaged_orbit_fixtures

        fixtures = verify_packaged_orbit_fixtures()
        thrust = finite_burn_report()
        covariance = covariance_prediction_report()
        geometry = geometry_event_report()
        result = {"schema": "space-env.physics-qualification/1",
                  "passed": fixtures["passed"] and thrust["passed"] and covariance["passed"] and geometry["passed"],
                  "qualification_status": "needs_external_evidence",
                  "orbit_fixtures": fixtures, "finite_burn": thrust, "covariance_prediction": covariance, "geometry_events": geometry}
        atomic_write(args.output, canonical_json(result))
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return

    if args.command in {"mission-validate", "mission-run", "mission-resume", "mission-replay", "mission-schema"}:
        from space_env.runtime.cli import run_command
        from space_env.runtime.mission import RuntimeContractError

        try:
            result = run_command(args)
        except RuntimeContractError as exc:
            print(json.dumps(exc.as_dict(), ensure_ascii=False))
            raise SystemExit(1) from exc
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command in {"reference-benchmark", "reference-verify-benchmark"}:
        from pathlib import Path

        from space_env.reference.benchmark import (
            evaluate_calendar_suite,
            save_calendar_benchmark,
            verify_calendar_benchmark,
        )

        if args.command == "reference-verify-benchmark":
            result = verify_calendar_benchmark(args.record)
            print(json.dumps(result))
            if not result["ok"]:
                raise SystemExit(1)
            return
        if Path(args.suite).resolve() == Path(args.output).resolve():
            parser.error("suite and output paths must differ")
        suite = json.loads(Path(args.suite).read_text())
        if set(suite) != {"cases"}:
            parser.error("benchmark suite requires exactly cases")
        result = evaluate_calendar_suite(suite["cases"], max_states=args.max_states, total_states=args.total_states)
        save_calendar_benchmark(args.output, result)
        print(json.dumps({"sha256": result["sha256"], "summary": result["summary"]}))
        return
    if args.command == "reference-optimal":
        from pathlib import Path

        from space_env.reference.optimal import solve_small_downlinks
        from space_env.simulation.snapshot import atomic_write, canonical_json

        paths = [Path(p).resolve() for p in (args.spec, args.output, args.report)]
        if len(set(paths)) != 3:
            parser.error("oracle spec, plan and report paths must differ")
        result = solve_small_downlinks(json.loads(paths[0].read_text()), max_states=args.max_states)
        atomic_write(paths[2], canonical_json(result))
        atomic_write(paths[1], canonical_json(result["plan"]))
        print(json.dumps({"scope": result["scope"], "on_time_bits": result["report"]["on_time_bits"],
                          "regret_bits": result["regret_bits"], "states": result["states_explored"]}))
        return
    if args.command in {"reference-events", "reference-adapt"}:
        from pathlib import Path

        from space_env.data_generators.reference_operations import (
            ReferenceOperationsRecorder,
        )
        from space_env.reference.adaptation import adapt_reference_run
        from space_env.simulation.snapshot import atomic_write, canonical_json

        paths = [Path(p).resolve() for p in (args.record, args.events, args.output)]
        if args.command == "reference-adapt":
            paths.append(Path(args.report).resolve())
        if len(set(paths)) != len(paths):
            parser.error("event input and output paths must differ")
        payload = json.loads(paths[1].read_text())
        if set(payload) != {"events"}:
            parser.error("events file requires exactly an events list")
        if args.command == "reference-adapt":
            recorder, schedule = adapt_reference_run(paths[0], payload["events"],
                                                     decision_sec=args.decision_sec, max_steps=args.max_steps)
            atomic_write(paths[3], canonical_json(schedule))
        else:
            recorder = ReferenceOperationsRecorder.from_record(paths[0])
            recorder.inject(payload["events"])
        record = recorder.save(paths[2])
        print(json.dumps({"sha256": record["sha256"], "report": record["final"]}))
        return
    if args.command in {"reference-plan", "reference-replan"}:
        from pathlib import Path

        from space_env.data_generators.reference_operations import (
            ReferenceOperationsRecorder,
        )
        from space_env.reference.scheduling import (
            plan_downlinks,
            plan_remaining_downlinks,
        )
        from space_env.simulation.snapshot import atomic_write, canonical_json

        paths = [Path(p).resolve() for p in (args.spec if args.command == "reference-plan" else args.record,
                                            args.output, args.report)]
        if len(set(paths)) != 3:
            parser.error("spec, plan and report paths must differ")
        if args.command == "reference-plan":
            result = plan_downlinks(json.loads(paths[0].read_text()), select_products=args.select_products,
                                    decision_sec=args.decision_sec, max_steps=args.max_steps)
        else:
            recorder = ReferenceOperationsRecorder.from_record(paths[0])
            result = plan_remaining_downlinks(recorder.mission, decision_sec=args.decision_sec,
                                              max_steps=args.max_steps)
        atomic_write(paths[2], canonical_json(result))
        atomic_write(paths[1], canonical_json(result["plan"]))
        print(json.dumps({"verified": result["verified"], "steps": len(result["plan"]["steps"]),
                          "on_time_bits": result["report"]["on_time_bits"],
                          "unfulfilled": result["unfulfilled"]}))
        return
    if args.command in {"reference-run", "reference-replay", "reference-resume"}:
        from pathlib import Path

        from space_env.data_generators.reference_operations import (
            ReferenceOperationsRecorder,
            replay_reference_operations,
        )

        if args.command == "reference-replay":
            result = replay_reference_operations(args.record)
            print(json.dumps(result))
            if not result["ok"]:
                raise SystemExit(1)
            return
        source_path = args.record if args.command == "reference-resume" else args.spec
        if len({Path(p).resolve() for p in (source_path, args.plan, args.output)}) != 3:
            parser.error("source, plan and output paths must differ")
        plan = json.loads(Path(args.plan).read_text())
        if ("steps" not in plan or set(plan) - {"steps", "initial_state_sha256", "start_sec"}
                or not isinstance(plan["steps"], list)):
            parser.error("reference plan requires a steps list")
        if args.command == "reference-resume":
            recorder = ReferenceOperationsRecorder.from_record(args.record)
        else:
            recorder = ReferenceOperationsRecorder(json.loads(Path(args.spec).read_text()), lineage=args.lineage)
        from space_env.reference.scheduling import mission_fingerprint

        if (("initial_state_sha256" in plan or "start_sec" in plan)
                and (plan.get("initial_state_sha256") != mission_fingerprint(recorder.mission)
                     or plan.get("start_sec") != recorder.mission.clock)):
            parser.error("continuation plan does not match the current mission state")
        for step in plan["steps"]:
            if not {"dt_sec", "actions"} <= set(step) or set(step) - {"dt_sec", "actions", "products", "processing", "processing_jobs"}:
                parser.error("each reference step requires dt_sec, actions and optional products/processing")
            recorder.step(step["dt_sec"], step["actions"], products=step.get("products"), processing=step.get("processing"),
                          processing_jobs=step.get("processing_jobs"))
        record = recorder.save(args.output)
        print(json.dumps({"sha256": record["sha256"], "report": record["final"]}))
        return
    if args.command == "reference-access":
        from pathlib import Path

        from space_env.core.ground import GroundStation
        from space_env.reference.access import ReferenceEphemeris
        from space_env.simulation.snapshot import atomic_write, canonical_json, digest

        raw = json.loads(Path(args.stations).read_text(encoding="utf-8"))
        if set(raw) != {"stations"} or not isinstance(raw["stations"], list) or not 1 <= len(raw["stations"]) <= 128:
            parser.error("stations file requires a stations list with 1-128 items")
        stations = []
        for row in raw["stations"]:
            if not isinstance(row, dict) or set(row) - set(GroundStation.model_fields):
                parser.error("unsupported station fields")
            stations.append(GroundStation.model_validate(row))
        if len({station.name for station in stations}) != len(stations):
            parser.error("station names must be unique")
        ephemeris = ReferenceEphemeris(json.loads(Path(args.orbit).read_text(encoding="utf-8")))
        body = {"schema": "space-env.reference-ground-network/1", "ephemeris_sha256": ephemeris.sha256,
                "stations": [ephemeris.access(station, scan_sec=args.scan) for station in stations]}
        result = {**body, "sha256": digest(body)}
        atomic_write(args.output, canonical_json(result))
        print(json.dumps({"stations": len(stations), "windows": sum(len(row["windows"]) for row in body["stations"]),
                          "sha256": result["sha256"]}))
        return
    if args.command == "physics-reference":
        from space_env.validation.gcrs_report import reference_physics_report

        result = reference_physics_report(long_arcs=args.long, crosscheck=args.crosscheck)
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "reference-propagate":
        from pathlib import Path

        from space_env.reference import (
            EOPSeries,
            Epoch,
            ForceModel,
            FrameContext,
            ReferenceState,
            propagate,
        )
        from space_env.simulation.snapshot import atomic_write, canonical_json, digest

        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        required = {"epoch_utc", "frame", "r_m", "v_m_s", "duration_sec"}
        optional = {"forces", "sample_sec", "integrator"}
        if not required <= set(spec) or set(spec) - required - optional:
            parser.error("reference spec requires epoch_utc/frame/r_m/v_m_s/duration_sec; optional forces/sample_sec/integrator")
        controls = spec.get("integrator", {})
        if set(controls) - {"rtol", "position_atol_m", "velocity_atol_m_s", "max_step_sec"}:
            parser.error("unsupported reference integration controls")
        initial = ReferenceState(Epoch.utc(spec["epoch_utc"]), spec["r_m"], spec["v_m_s"], frame=spec["frame"])
        result = propagate(initial, spec["duration_sec"], FrameContext(EOPSeries.packaged(allow_predictions=args.allow_predictions)),
                           model=ForceModel(**spec.get("forces", {})), sample_sec=spec.get("sample_sec", 60.), **controls)
        record = {**result, "sha256": digest(result)}
        atomic_write(args.output, canonical_json(record))
        print(json.dumps({"profile": result["manifest"]["profile"], "frame": result["frame"],
                          "samples": len(result["times_sec"]), "sha256": record["sha256"]}))
        return
    if args.command == "record-policy":
        from space_env.data_generators.agent_run import record_baseline
        from space_env.mission.tools import MissionTools

        result = record_baseline(MissionTools(args.scenario, seed=args.seed, call_budget=args.call_budget),
                                 args.output, lineage=args.lineage, policy=args.policy,
                                 step_sec=args.step, max_steps=args.max_steps)
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command in {"record-tools", "replay-tools", "export-dialogue"}:
        from pathlib import Path

        from space_env.data_generators.agent_run import (
            AgentRunRecorder,
            export_tool_dialogue,
            replay_agent_run,
        )
        from space_env.mission.tools import MissionTools

        if args.command == "replay-tools":
            result = replay_agent_run(args.record)
        elif args.command == "export-dialogue":
            result = export_tool_dialogue(args.record, args.output)
        else:
            script = json.loads(Path(args.script).read_text(encoding="utf-8"))
            if set(script) != {"task", "calls"} or not isinstance(script["calls"], list):
                parser.error("tool script requires exactly task and a calls list")
            tools = MissionTools(args.scenario, seed=args.seed, call_budget=args.call_budget)
            recorder = AgentRunRecorder(tools, task=script["task"], lineage=args.lineage,
                                        provenance={"interface": "record-tools", "teacher": "script", "expert": False})
            for call in script["calls"]:
                if not {"name", "arguments"} <= set(call) or set(call) - {"name", "arguments", "call_id"}:
                    parser.error("call requires name, arguments and optional call_id")
                recorder.invoke(call["name"], call["arguments"], call_id=call.get("call_id"))
            record = recorder.save(args.output)
            result = {"run_sha256": record["sha256"], "events": len(record["events"]),
                      "failed_calls": sum(not event["response"]["ok"] for event in record["events"])}
        print(json.dumps(result, ensure_ascii=False))
        if result.get("ok") is False:
            raise SystemExit(1)
        return
    if args.command in {"record-commands", "replay-run", "export-transitions"}:
        from pathlib import Path

        from space_env.data_generators.portable_run import (
            RunRecorder,
            export_transitions,
            replay_run,
        )
        from space_env.mission.scenario_v2 import load_scenario_v2, world_from_scenario
        from space_env.mission.tasking import requests_from_scenario
        from space_env.simulation.commands import MissionCommand

        if args.command == "replay-run":
            result = replay_run(args.record)
        elif args.command == "export-transitions":
            result = export_transitions(args.record, args.output)
        else:
            plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            if set(plan) != {"steps"} or not isinstance(plan["steps"], list):
                parser.error("plan must be an object with a steps list")
            spec = load_scenario_v2(args.scenario)
            world = world_from_scenario(args.scenario, attach_stations=True, refresh_passes=False)
            world.task_requests = {request.request_id: request for request in requests_from_scenario(spec)}
            world.set_outages([(row["station"], row["start_sec"], row["end_sec"]) for row in spec["outages"]])
            recorder = RunRecorder(world, lineage=args.lineage, provenance={"interface": "record-commands"})
            for step in plan["steps"]:
                if set(step) != {"dt_sec", "commands"}:
                    parser.error("each step requires exactly dt_sec and commands")
                recorder.step(step["dt_sec"], [MissionCommand(**row) for row in step["commands"]])
            record = recorder.save(args.output)
            result = {"run_sha256": record["sha256"], "steps": len(record["steps"])}
        print(json.dumps(result, ensure_ascii=False))
        if result.get("ok") is False:
            raise SystemExit(1)
        return
    if args.command == "validate":
        from space_env.mission.scenario_v2 import load_scenario_v2

        spec = load_scenario_v2(args.scenario)
        print(json.dumps({"instances": [item["id"] for item in spec["instances"]], "requests": [item["id"] for item in spec["requests"]]}, ensure_ascii=False))
        return
    if args.command == "benchmark":
        from space_env.eval.mission_benchmark import run_mission_suite

        print(json.dumps(run_mission_suite(args.suite), ensure_ascii=False))
        return
    if args.command == "replay":
        from space_env.data_generators.mission_dataset import replay

        print(json.dumps({"ok": replay(args.record)}, ensure_ascii=False))
        return
    report = mission_report(args.scenario, step_sec=args.step, policy=args.policy)
    print(
        json.dumps(
            {
                "on_time": report["on_time"],
                "unmet": report["unmet"],
                "weighted_on_time": report["weighted_on_time"],
                "safety_violations": report["safety_violations"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
