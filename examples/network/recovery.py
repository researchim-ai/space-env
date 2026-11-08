"""Reproduce native custody rerouting after a receiver fault (synthetic)."""

from space_env.api.network_workspace import default_network_configuration
from space_env.simulation.snapshot import digest
from space_env.workspace import MissionWorkspace


def main():
    config = default_network_configuration()
    config["mission"]["planes"][0]["phases_deg"].append(40.0)
    config["links"].append(
        {"source": "fleet-01", "destination": "fleet-03", "max_range_m": 5000000.0}
    )
    source = MissionWorkspace.network(config)
    branch = None
    try:
        source.run(decisions=22)
        before = digest(source.operator.session.checkpoint(compact=True))
        branch = source.evolve(
            "Обойти отказавший приёмник",
            idempotency_key="receiver-fault",
            events=[
                {
                    "type": "spacecraft_fault",
                    "payload": {"actor": "fleet-02", "subsystem": "radio"},
                }
            ],
        )
        proposed = branch.state()["network"]
        print("Current custodian:", proposed["dynamic"]["states"]["survey"]["owner"])
        print("Remaining route:", proposed["requests"][0]["plan"]["path"])
        result = branch.run()["network"]
        unchanged = digest(source.operator.session.checkpoint(compact=True)) == before
        print(f"Native ACK: {result['fulfilled']}/{result['total']}")
        print("Source checkpoint unchanged:", unchanged)
        return 0 if unchanged and result["fulfilled"] == result["total"] else 2
    finally:
        if branch is not None:
            branch.close()
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
