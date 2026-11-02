# B05-B — concurrent reference ISL and bounded custody transport

2026-11-02; package 0.134.0. This engineering package connects directed RF links,
endpoint resources and event-driven custody to the existing reference runtime and
managed product lifecycle. It is **not** BPv7/BIBE/LTP wire interoperability,
flight software, calibrated channel simulation or a fleet performance qualification.

## Configuration, control and compatibility

Opt in with `operations.dtn_network` for a dynamic/lifecycle/spacecraft mission,
or `definition.dtn_network` for a fixed-reference scenario. An explicit
`radio_network` is required: the same actor TX hardware and net-information modem
rates serve ground and ISL traffic. Configure receiver DC power, minimum SOC and
memory bandwidth for every actor, plus unique directed source/destination links.
Each link declares RF receiver G/T/band/noise/margin, optional receiver BODY
beamwidth, acquisition time, maximum range, synthetic data/ACK loss and outages.
Ground aggregate/setup/ACK policies cannot silently be reused as ISL policies.

Declare a source and synthetic/published/calibrated parameter status. These are
caller provenance, not certification. Limits cover actors, directed links,
transfers, fragments, histories, intervals and total scheduled events. Exhaustion
fails atomically. Completed transfer IDs are immutable and count toward the budget.
A disabled DTN configuration preserves the previous direct ground behavior.

Submit a normal runtime command:

```json
{"command_id":"hop-1","actor_id":"source","kind":"relay",
 "duration_sec":10.0,"destination_id":"next-hop","product_id":"product",
 "relay":{"ttl_sec":20.0}}
```

The control window and the data lifetime are separate. The caller chooses a
next hop; no hidden optimal route or future-fault oracle is inserted. Products
must be intact, processed, locally owned and unretained by future native jobs.
DTN requires an exact integer bit count within 2^53. Managed lifecycle transport
currently accepts **final deliverables only**; moving raw/pinned intermediates to
remote CPU workers is a separate distributed-compute extension.

The sender's command reserves its normal runtime lanes. With resource lanes,
relay conservatively owns the sender's CPU/payload/radio/attitude/propulsion
control lanes. Passive receivers admit data autonomously when their local command
resources permit it. Existing local controls take priority over a queued incoming
transfer. Disjoint endpoint pairs and disjoint direct/relay traffic execute on
the same mission clock; the implementation is not a blocking whole-transfer call.
A finished transport does not shorten its caller's control window automatically.

## RF, geometry and half-duplex admission

The same independently tested Friis/EIRP/G/T/C/N0/information-Eb/N0 kernel selects
feasible discrete modes. Geometry uses both actual GCRS providers on an absolute
sample grid, checks configured maximum range and spherical Earth occultation with
an explicit limb altitude. Range-rate gives first-order Doppler. These are nominal states; navigation
covariance is not propagated into RF outage probability. No trajectory
extrapolation, future pending maneuver execution or perfectly known future loss
is inserted into observations.

Optional TX and RX patterns use actual BODY +Z orientations; directional pointing
is never created by a relay command. An agent must arrange attitude before its
exclusive relay window. Defaults explicitly assume ideal pointing. The receive
pattern is a bounded parabolic main-lobe surrogate, not a measured antenna map.
Interference, beam sidelobes, polarization rotation and simultaneous-channel
frequency coordination are not simulated. Disjoint links assume spatial/channel
isolation. This is a reduced RF engineering model, not Maxwell/waveform simulation.

An interval allocator grants each endpoint at most one radio role. Active packet
serialization/acquisition precedes queued work, then acquisition state and stable
transfer ID break ties. A common receiver queues contenders rather than doubling
its throughput or power. No optimal/fair scheduler or starvation guarantee is
claimed. Native downlink, CPU work and managed work at a receiver prevent an
incoming packet from starting. Radio health, safe mode, shared bus peak, battery
reserve, native TX quota/thermal lock and memory bandwidth are admission gates.

## Storage, serialization, propagation and energy

Before acquisition the destination reserves **the full product size** in its real
memory capacity. This includes received staging bytes plus the remaining reserved
space; those are not counted twice. Existing products, native workspace, managed
job/acquisition reservations and competing custody reservations remain occupied.
Insufficient memory produces queue backpressure without silent loss or allocation.
New product injection cannot spend a custody reservation. Source data is pinned
against native deletion or concurrent ground transmission while custody is unsettled.

The packet model uses fixed maximum fragment payload bits and stop-and-wait ARQ.
Each attempt pays configured acquisition time with source TX and destination RX
power. Actual payload serialization then advances at an admitted net-information
rate. Native TX seconds include acquisition; application-bit traffic excludes it.
No unmodeled packet-header bytes or BP framing are asserted. Existing modem
payload fraction is the declared coding/framing efficiency.

Once a fragment finishes serialization, its event is scheduled at
`serialization_end + sampled_range/c`. It is not staged at the receiver before
arrival. The receiver listens through the propagation tail and retains its radio
role until packet arrival, even after source TX has stopped. This is a conservative
listening model, not a delayed continuous waveform sample stream.

Source and receiver DC/SOC are integrated separately by the existing resource
executor. Shared bus peak and existing heater reserves constrain both endpoints.
RX DC energy is counted explicitly and routed through the coupled thermal radio
source; conversion losses keep their existing route. Legacy transmitter-only RC
models still describe TX, not an invented receive thermal calibration. Interrupting
a packet preserves already consumed energy and counts partial serialized bits as
waste; subsequent transmission restarts the fragment rather than resuming a broken
coded packet. No power, fuel, temperature, storage or momentum resets occur.

## Event queue, losses, retries and custody

Events have deterministic (time, sequence) order on SI mission seconds. Boundaries
include acquisition, packet completion, propagation arrival, ACK/timeout, retry,
outage, control end and lifetime. Actual radiative/orbital/resource forcing remains
sampled as declared by the existing profiles. Different caller dt partitions can
change trace segmentation and last-place floating-point bits; checkpoints with
identical operation journals replay exactly.

Data and ACK loss use SHA256-keyed Bernoulli samples from declared seed, transfer,
fragment and complete-attempt number. Outcomes are realized at arrival/ACK, not
published in advance. Interrupted partial attempts count separately as waste.
An outage during the receive tail invalidates the packet. A successful fragment
gets an ideal out-of-band ACK after return light time and configured processing.
Endpoint health also gates confirmation. Missing ACK triggers bounded fixed-delay
retry after an explicit timeout; retries retain the same fragment identity within
the hop/session. Duplicate fragments consume TX/RX resources but allocate no more
unique staging bytes and create no extra custody/product credit.

ACK is deliberately a reduced control-channel abstraction: no reverse RF bandwidth,
ACK packet energy, contention, encoding, cryptography, authentication, telemetry
latency or actual hardware decoder is modeled. The controller observes ideal
centralized current/past transport state. This does not establish delayed
information mechanics for decentralized MARL. Scenario seeds are not a security
boundary against an agent with filesystem access.

Source native bytes remain intact until **every fragment has arrived and been
acknowledged**. At that event the native product ID/content is preserved and its
transport owner changes atomically. Receiver staging becomes native inventory;
the source custody copy is released once. Managed artifacts retain `origin_actor`
and acquisition/observation provenance while `actor` tracks storage ownership.
The receiver can now forward or downlink the product; it could not use it early.
Managed raw/intermediate products cannot bypass retained inputs or workspace pins.

This is an explicit application-level custody handshake. [RFC 9171](https://www.rfc-editor.org/rfc/rfc9171.html)
describes bundle identity, lifetime and reassembly; it moves the older custody
mechanism out of the BP core. No claim of BPv7 custody signalling, CBOR/BIBE/LTP
compatibility or standards qualification follows from this simulator.

## Lifetime, cancellation, rerouting and exactly-once ground credit

The first transport binds an absolute product/bundle expiry. A later hop or reroute
may shorten it but never extend it. Existing managed artifact TTL can shorten it
as well. At expiry, owned bytes and all staging copies become unusable, native
inventory is discarded and managed requests fail/expire consistently. In-flight
payload is recorded as abandoned, not converted into useful inventory or refunded
energy. Confirmation exactly at expiry loses to the lifetime boundary. An already
confirmed ground delivery remains confirmed after its former bundle lifetime.

Cancellation during partial serialization discards the partial packet, releases
staging and preserves the source copy. Cancellation with an in-flight fragment or
pending ACK drains the attempt through ACK/timeout before releasing its reservation.
It cannot undo completed custody. Rerouting uses a new transfer command after the
old transfer has settled; it preserves absolute lifetime and data identity. Native
product cancellation/deadline mutation while custody is pinned is rejected.

A custody ACK is **not ground delivery**. Native receipts and managed customer
credit still require final station serialization and the B05-A complete-product
return ACK. TTL, request deadline, cancellation and quality/coverage constraints
remain active. Retries, duplicate packets, repeated waits and cold replay cannot
create a second native product or duplicate completed-request credit.

## Independent balances and rollback

Three different quantities remain separate:

1. Logical native inventory per actor: initial + injected + custody in - custody
   out - ground serialized - discarded - remaining. Custody history records each
   owner move once, without pretending it is newly acquired data.
2. Receiver staging: unique received = committed custody + discarded staging +
   currently staged unique bits. Reservations bound actual allocation separately.
3. Payload traffic: total TX serialized = unique received + duplicate received +
   lost complete packets + wasted partial packets + abandoned in-flight packets +
   current in-flight payload + current partial serialization.

Residuals are reported from independent counters; no missing term is defined as
whatever is needed to force zero. Ground ACKed complete products are separate from
raw wire traffic. Each actor also retains existing native battery and coupled
thermal conservation ledgers. Acquisition and propagation listening cost energy
even when no customer product is eventually delivered.

Native step rollback restores owner records, remaining bytes, cancellations,
discard counters, RX/TX energy/time, reservations, packet event queue and histories.
Runtime rollback also preserves shared mission/lifecycle/transport/bus aliases.
Invalid inputs, stale IDs, bounded event/history failure and receiver-memory
conflicts do not leave a half-committed custody transfer.

## Evidence and remaining industrial gates

44 new parametrized cases cover packet causality, ownership/ACK timing, endpoint
power and propagation listening, shared receiver scheduling, independent concurrent
pairs, mixed ground/ISL, buffer pressure, data/ACK loss, duplicate packets, retry
cost, partial interruption, tail outage, TTL and multi-hop non-extension, cancellation
and draining, spherical occultation/range, receive BODY pointing, SOC/peak/quota,
managed thermal protection and origin provenance, exact cold replay, operator pins,
budget/schema rejection and native/runtime rollback including expiry/discard.

Run `python examples/relay_mission/demo.py OUTPUT nominal|reroute|loss|ttl`.
These are **three-actor EO missions with no preloaded payload data**:

- nominal: source capture/CPU, custody to C, second capture, concurrent direct C→ground
  and A→B relay, then B→ground; 2/2 customer requests.
- reroute: interrupted first ISL packet, ground outage, full/insufficient B buffer,
  queued second hop, cancellation and reroute through C; 2/2.
- loss: a declared deterministic lost ACK, paid retry and receiver deduplication;
  2/2 without duplicate customer credit.
- ttl: second bundle expires while work is incomplete; 1/2 with no false completion.

Prefix, actual in-flight fragment and final checkpoints are cold replayed. Results,
installed SDK/CLI checks and final regression are recorded separately in
`docs/acceptance/B05_CUSTODY_NETWORK.json`. Synthetic parameter provenance remains
explicit; published/calibrated input status alone does not validate an instrument.

B05-A/B together close the declared B05.01–10 **bounded engineering profile**.
Industrial completeness is not implied: multi-radio/multi-band terminals,
streaming/cut-through or partial-bundle forwarding, remote managed CPU, multipath
replication/multicast custody, calibrated RF/interference/weather, physical ACK,
protocol interoperability, distributed delayed observations, B09 fleet evidence
and B12 independent operator/hardware qualification remain open extensions.
