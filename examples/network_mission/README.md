# RF/ground mission quickstart

Requires the reference and coverage extras used by the EO/thermal example:
`pip install -e '.[reference,coverage]'`.

```bash
python examples/network_mission/demo.py /tmp/network-nominal nominal
python examples/network_mission/demo.py /tmp/network-outage outage
python examples/network_mission/demo.py /tmp/network-blocked blocked
```

Each run builds a scenario and plans, captures imagery, runs a managed transform,
requests reserved ground service, serializes products and waits for confirmation.
It saves `prefix.json` and `transmitting.json` during the first radio command,
and `final.json` at the end. In nominal mode the prefix is mid-acquisition. Each
checkpoint is verified by cold prefix replay. The outage prefix is before station
recovery; its named transmission checkpoint can still be acquiring. In blocked
mode neither checkpoint implies that a carrier has locked or payload bits flowed.

Expected outcomes:

- `nominal`: 2/2 requests fulfilled after ground setup and delayed ACK.
- `outage`: 2/2 after station recovery, lost lock and paid reacquisition.
- `blocked`: 0/2; the RF margin does not close. No payload bits or ground setup
  energy are credited. Capture/compute still cost satellite resources.

All spacecraft/RF/ground/weather parameters are synthetic engineering inputs.
The demo includes a five-node thermal network, battery limits, CPU work, an
imported versioned ground reservation and shared station capacity limits.

## CLI continuation

After a demo has generated the files:

```bash
space-env mission-validate /tmp/network-nominal/scenario.json
space-env mission-run /tmp/network-nominal/scenario.json /tmp/network-nominal/plan.json /tmp/network-nominal/cli-prefix.json
space-env mission-resume /tmp/network-nominal/cli-prefix.json /tmp/network-nominal/resume.json /tmp/network-nominal/cli-final.json
space-env mission-replay /tmp/network-nominal/cli-final.json
```

SDK and CLI final checkpoints are tested for exact equality. Preserve the source
and dependency versions recorded by the runtime; fingerprints are not bypassed.

## Reading the evidence

A saved control-plane checkpoint contains `backend_report`:

- `radio_network.links`: last sampled range, Doppler, EIRP, C/N0, modem candidates,
  chosen mode, occupied bandwidth and acquisition/lock state.
- `radio_network.transitions`: terminal acquisition/release history, with ready
  times. A visible contact alone is not a working link.
- `radio_network.intervals`: allocations and actual station bits per interval;
  aggregate allocations stay within configured limits.
- `radio_network.ground_energy_j`: external ground-station supply energy;
  satellite DC/thermal costs are separately under `spacecraft_bus.ledgers`.
- `receipts`: serialized fragment intervals plus ideal return-channel ACK due
  times. These receipts do not themselves grant complete-product reward.
- `lifecycle.acknowledgements` and `lifecycle.metrics`: confirmed products and
  customer fulfillment. Late or incomplete confirmation is not on-time success.
- `external_reservations`, `bookings`, `events`: imported revisions, local
  capacity allocations and causal operator history.

Agent observations expose current/past RF evidence; the full checkpoint is a
truth/control-plane artifact subject to runtime access policy. Link evidence is
sampled, not a prediction of future weather or guaranteed future throughput.

See the [RF/ground contract](../../docs/validation/RADIO_GROUND_CONTRACT.md) and
[acceptance evidence](../../docs/acceptance/B05_NETWORK.json). Concurrent reference ISL/custody and packet loss/retry are implemented separately
in [B05-B](../relay_mission/README.md). Interference, calibrated rain and physical
ACK resource contention remain open gates.
