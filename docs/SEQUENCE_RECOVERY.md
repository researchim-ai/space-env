# Sequence-aware dynamic recovery

`network_dynamic` replays the actual committed controls into an isolated branch,
cancels active commands through native execution, applies mission events, and
then calls `planning.sequence_recovery.schedule_recovery`. The source session
remains unchanged. Historical scenario definitions are not rewritten.

The planner uses two bounded stages:

1. Existing `joint_network` recovery schedules raw/final products from their
   observed custodian. It retains native remaining CPU work, terminal states,
   product/bundle expiry and partial final-product custody restrictions. Raw
   products must be intact and have settled custody. Recovery never reacquires
   an already acquired request.
2. Fresh requests enter a separately sealed `sequence_lanes` forecast after
   every admitted residual allocation finishes. Its initial quaternion is the
   **committed native quaternion after cancellation/events**, not the original
   scenario attitude. Capture goals are newly queried from that runtime; native
   command claims, CPU job power, lane ownership, peak load and memory bandwidth
   are collected again. Current stored/reserved bits plus conservative residual
   memory costs and residual energy costs reduce the suffix capacities. The two
   searches share the configured node budget.

The supported attitude profile is nominal legacy rate-limited kinematics with
omnidirectional downlink, no current safe mode/faults, and native CPU, payload,
attitude and exclusive relay lane claims. Unresolved settling delays the suffix.
Residual transform/relay/omnidirectional downlink work does not change the
committed attitude, so it can be held until the fresh suffix starts without
executing a hypothetical future. Within that fresh suffix, the existing sequence
planner can overlap CPU work and later attitude work when native lanes, power,
bandwidth and command-start constraints permit it.

Unsupported rigid attitude, degraded slew rate, directional radio, faults,
unsettled commands, unavailable conservative capacities, exhausted shared search
budget or absence of a suffix before a fresh deadline produces an explicit
`conservative_fallback`. The original complete recovery view is replanned with
`joint_network.schedule_network`, including fresh requests and all usable actors.
No sequence certificate is retained. The remaining shared node budget bounds
that serial fallback; if exhausted, the existing residual incumbent is retained
and fresh work is explicitly unserved for search-budget exhaustion. A fault on
an unused actor or an early fresh deadline can prevent sequence admission, but
does not itself remove that request from serial recovery. Serial recovery is a
conditional conservative proposal, not complete recovery for arbitrary native
spacecraft models.

`network.dynamic.sequence_recovery` exposes the mode, fallback reason and fresh
request IDs. Internal metadata `sequence_recovery` stores the separate suffix
forecast, geometry and verified schedule (null on fallback). Geometry includes
committed time and physical-report digest. Original `sequence_geometry`,
`pointing_envelope`, completeness and terminal geometry/base schedule certificates
are removed. `planning_method` is `sequence_recovery` for an admitted fresh
sequence suffix and `serial_recovery` for residual-only or fallback plans. The
original `sequence_lanes` UI summary is not reused for this hybrid plan.

The resulting schedule is a conditional proposal, with `optimal: false`. Memory
is reserved for the lifetime of the proposal rather than reclaimed after
transfer, partial ground products keep a conservative full-product transmission
reservation, and residual work has priority over all fresh work regardless of a
new fresh request's priority. There is no global optimum, arbitrary residual /
fresh lane interleaving, or guaranteed native success. Native execution remains
the final acceptance test.

Targeted verification is in `tests/test_sequence_recovery.py` and the existing
`tests/test_network_dynamic.py`: changed native attitude and fresh scheduling
after repriority; partial CPU/raw work and partial final ground transmission
alongside event-added jobs; preserved content hashes, one acquisition per job,
actual delivery acknowledgements, source immutability, checkpoint replay,
explicit fallback, and geometry collection without submit/advance calls.
