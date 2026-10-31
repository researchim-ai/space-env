# Reference service lifecycle: ten-task package

Update: [Ground booking v5](REFERENCE_GROUND_BOOKING_V5.md) adds in-process terminal
booking, conflict checks, admission policies, window diagnostics and utilization.
External ground-network integration and antenna setup remain open.

Implemented 2026-10-31 over the separate fixed-ephemeris operations profile.
This is an executable operational model extension, not an RF-calibrated flight
simulator or a production/scale qualification. No release version bump or push.

## Ten completed tasks

1. **Product cancellation.** `product_cancelled` event payload is `{product_id}`.
   Only unresolved inventory can be cancelled. Already delivered bytes and
   receipts remain; undelivered bytes are removed from onboard memory. Product
   identity stays allocated and cannot be reused. Cancellation is irreversible
   in this model, recorded with time/source/reason and participates in atomic
   ordered batch validation. Selecting a cancelled product subsequently waits.
2. **Discard accounting.** Every product report has discarded_bits, cancelled_sec
   and a distinct cancelled status. Inventory balance is now
   initial + injected = delivered + remaining + discarded. Partial transmission
   followed by cancellation never masquerades as full delivery. Cancellation
   can free memory for a replacement product in the same ordered event batch.
3. **Deadline changes.** `product_deadline_changed` payload is product_id and
   deadline_sec. The product must be unresolved and the new deadline must be
   later than both now and readiness, within the horizon. Delivered outcomes
   cannot be retrospectively reclassified. Extending an expired unresolved
   deadline is allowed as an explicit operator requirement change, not hidden
   reward relabeling. Earlier receipts and discarded data are never rewritten.
4. **Ground station throughput limit.** `station_rate_changed` payload station
   and finite nonnegative rate_bps. It is a ceiling **per active terminal**, not
   aggregate bandwidth sharing. Zero blocks throughput; positive updates can
   restore it. Initial ceiling is absent. This is an injected service constraint,
   not a physical antenna/RF/link-budget calculation.
5. **Transmitter degradation and restoration.** `transmitter_rate_changed`
   payload actor and finite nonnegative rate_bps. Values cannot exceed nominal
   VehicleResources.rate_bps; zero disables transmission. Restoring the nominal
   value is explicit. Actual rate is min(current transmitter rate, station cap).
   Blocked actors consume base power only and do not occupy a transmitting
   terminal. For positive rates the configured TX power remains constant; no
   power/rate adaptation law or noise model is implied.
6. **Planner integration.** Slack, duration and candidate feasibility use actual
   effective throughput. Zero-rate pairs are skipped without division by zero.
   Cancelled products are absent from candidates and reported separately as
   cancelled_by_operator, not an unexplained planner failure. Baseline remains
   privileged, greedy and not proven optimal.
7. **RL integration.** ReferenceEventDownlinkEnv exposes slot_cancelled,
   transmitter_rates_bps and station_rates_bps with stable observation shapes.
   The latter is clamped to the fleet's largest nominal transmitter rate because
   larger station capacity cannot increase any pair's throughput. The exact
   configured station cap remains in report.rate_limits. Cancellation resolves
   an item for termination, but grants no reward and makes whole-episode success
   false. This avoids treating cancelled work as successful delivery. Product
   slots remain allocated after cancellation; no observation/action index reuse.
8. **Replay/continuation integration.** The new event types use the existing /2
   journal and from_record restoration. Cancellation, rates, deadlines, discard
   accounting and source history must reexecute identically. State fingerprints
   include events and rate/report outcomes, invalidating plans made before a
   lifecycle change. Exact runtime matching still prevents unqualified journal
   migration across revisions.
9. **CLI workflow and scenario.** Existing reference-events/reference-adapt
   accept all new event types. `examples/reference_operations/lifecycle_demo.py`
   runs a two-satellite case with cancellation/replacement, changed deadline,
   degraded transmitter, ground throughput cap and preexisting station outage;
   saves the prefix, event file, verified schedule, portable run and RL JSONL.
10. **Numerical completion and regression coverage.** Fractional throughput
    exposed a sub-bit remainder at an absolute completion endpoint. Completion
    now allows at most four ulps of the represented time endpoint, rather than
    an arbitrary remaining-bits threshold. This avoids a no-progress failure
    for 0.75 Mbit/s while a meaningful partial transfer remains incomplete.
    Regression tests cover partial cancellation, batch rollback, memory reuse,
    lifecycle validation, zero-rate energy, multi-terminal per-terminal caps,
    planner/replay/CLI/Gym integration and fractional-rate completion. This does
    not qualify extreme throughput/time magnitudes outside measured scenarios.

## Use

```bash
python examples/reference_operations/lifecycle_demo.py /tmp/space-env-lifecycle
space-env reference-adapt /tmp/space-env-lifecycle/prefix.json \
  /tmp/space-env-lifecycle/events.json /tmp/space-env-lifecycle/cli-run.json \
  --report /tmp/space-env-lifecycle/cli-schedule.json
space-env reference-replay /tmp/space-env-lifecycle/cli-run.json
```

The demo delivers 3 Mbit on time, explicitly discards 3 Mbit of cancelled work,
and verifies replay. The RL rollout earns 3 Mbit reward, resolves all work and
reports success=false because one original product was cancelled. That is an
intentional distinction between termination and successful delivery.

## Next gates

Still required for broader industrial missions: causal acquisition and onboard
processing with resource costs; storage retention/expiry policy; aggregate RF
sharing, antenna setup/slew costs; station booking service and conflicts;
look-ahead energy/uncertainty; independent small-instance optimum and held-out
regret evaluation; large-fleet endurance; general mission backend integration.
No arbitrary admission of new satellites or station topology during an episode.
