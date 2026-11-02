# Concurrent custody mission

Install the existing reference/coverage extras, then run:

```bash
python examples/relay_mission/demo.py /tmp/relay nominal
python examples/relay_mission/demo.py /tmp/relay-reroute reroute
python examples/relay_mission/demo.py /tmp/relay-loss loss
python examples/relay_mission/demo.py /tmp/relay-ttl ttl
```

The first three modes fulfill 2/2 requests; TTL mode fulfills 1/2 and expires the
second product. Data is captured and processed during the mission, not preloaded.
A, B and C execute directed relay/ground operations on one clock with battery,
thermal, buffer and half-duplex constraints. In reroute mode B lacks enough buffer,
a packet is interrupted, a ground outage occurs, and the operator redirects through C.

Each directory contains `scenario.json`, initial `plan.json`, continuation
`resume.json`, `prefix.json` during a fragment, `inflight.json` before receiver
arrival, and `final.json`. Actual in-flight state is asserted before saving.

```bash
space-env mission-validate /tmp/relay/scenario.json
space-env mission-run /tmp/relay/scenario.json /tmp/relay/plan.json /tmp/relay/cli-prefix.json
space-env mission-resume /tmp/relay/cli-prefix.json /tmp/relay/resume.json /tmp/relay/cli-final.json
space-env mission-replay /tmp/relay/cli-final.json
```

A saved control-plane checkpoint contains `backend_report.dtn`: transfer states,
packet arrivals/ACK/retries, per-interval endpoint roles, custody moves, RX energy,
staging and wire balances. Native and lifecycle ledgers show custody in/out;
artifacts retain acquisition `origin_actor` while `actor` follows transport ownership.
Only `lifecycle.acknowledgements/metrics` show confirmed customer delivery.

All parameters are synthetic. Packet/ACK behavior is an explicit engineering
surrogate, not BPv7/BIBE/LTP or calibrated RF hardware. See the
[contract](../../docs/validation/CUSTODY_NETWORK_CONTRACT.md) and
[acceptance evidence](../../docs/acceptance/B05_CUSTODY_NETWORK.json).
