# B05-A — sampled RF and ground-terminal engineering profile

2026-11-02; package 0.134.0. This package connects a declared RF budget to actual
mission transport, storage, DC/thermal accounting, ground reservations and ACK.
B05-A alone does not execute concurrent ISL/custody. The later opt-in B05-B
package now covers that separately: see [custody contract](CUSTODY_NETWORK_CONTRACT.md).
Legacy relay classes alone do not establish that integration.

## Opt-in and compatibility

Configure `definition.operations.radio_network` for dynamic/lifecycle/spacecraft
missions, or `definition.radio_network` for a fixed-reference scenario. The
`sampled-rf-ground/1` model requires explicit source and parameter status
(`synthetic`, `published`, `calibrated`), one transmitter per actor and one receiver
per ground station. These are caller provenance, not independent certification.
Legacy `GroundStation.band_hz/share_bandwidth` flags remain rejected in reference
operations; configure bands/policies explicitly on the RF receivers instead.
Absent this field, legacy constant-rate admission and immediate receipt ACK stay
unchanged. No automatic replacement of legacy MissionWorld link calculations.

The model supports at most 64 actors, 128 stations, 64 terminals per station and
512 total terminals. Bounded RF cadence is 0.01–30 seconds; histories have explicit
limits. Exceeding a history budget fails atomically rather than deleting evidence.
The resource-lane executor preserves ground acquisition without falsely reserving
an active TX lane or repeatedly resetting lock on each internal interval.

## Independent budget and modem semantics

The pure kernel `reference.radio.rf_budget` implements:

- `FSPL_dB = 20*log10(4*pi*range_m*frequency_hz/c)` with SI `c=299792458`.
- `EIRP_dBW = 10*log10(RF_W) + gain_dBi - feeder_loss - fixed_pointing_loss - pattern_loss`.
- `C/N0_dBHz = EIRP - FSPL + G/T - polarization_loss - other_loss - fade - 10*log10(k)`;
  `k=1.380649e-23 J/K`. Receiver feeder/noise effects already included in G/T
  must not be charged again.
- Net information-bit `Eb/N0 = C/N0 - 10*log10(information_rate_bps)`.
  Margin subtracts caller-declared modem threshold and implementation loss.
- Occupied bandwidth is `R_info*(1+rolloff)/(bits_per_symbol*code_rate*payload_fraction)`.
  Friis assumes far-field line-of-sight propagation without multipath.
  Carrier RF power is fixed while choosing a mode; code/packet overhead does not
  create extra application throughput or get subtracted a second time.

Choose the highest feasible discrete net information rate, with stable ID tie
breaking. The entire occupied band must fit the receive passband and assigned
bandwidth; Doppler-shifted carrier must remain inside it and inside the declared
ideal tracking range. Doppler derives from current sampled GCRS/ITRS range rate.
Actor nominal/rate-event and station rate-event ceilings apply before modem
selection; e.g. a 750 kbps cap selects a feasible 500 kbps mode rather than inventing
a 750 kbps waveform. Modulation names/thresholds are not a DVB-S2/CCSDS decoder or
BER/PER curve. No Shannon-capacity, certification or spectrum licence claim.

RF output is bounded by actor TX DC power. Coupled buses additionally reject RF
output plus declared generic radio heat above DC input. Native transmitter RC heat
also cannot exceed DC together with RF. Ground energy comes from a separate
external station supply, never from the satellite battery. Existing thermal routing
accounts for actual TX time and declared heat; this is not a calibrated RF amplifier.

Source for the link-budget structure and definitions:
[MathWorks Satellite Link Budget](https://www.mathworks.com/help/satcom/gs/satellite-link-budget.html).
The implemented algebra is independently tested against linear Friis/noise-density
identities, not against copied output or a presumed external qualification.

## Geometry, pointing and environmental samples

Geometric access is independent from RF closure, booking eligibility, acquisition,
serialization and confirmation. Seeing a station never guarantees a working link.
Range/range-rate are held at the beginning of an absolute RF sample cell. Actual
execution also splits on resource, weather, access, booking, lock and complete-product
ACK boundaries. This is sampled quadrature, not continuous fading integration;
caller partitioning can add pointing samples and changes trace segmentation.

With no beamwidth, ideal transmitter pointing is explicit. With beamwidth enabled,
a coupled bus is required: actual BODY +Z antenna direction is compared with the
station line of sight. The bounded parabolic main-lobe surrogate is
`min(max_loss,12*(angle/full_HPBW)^2)` dB, so half the full beamwidth gives 3 dB.
There is no measured sidelobe, polarization attitude or deployment/ray-tracing model.
The default ground receive G/T is a declared constant, not an elevation noise map.

Optional per-station weather uses a SHA256-keyed deterministic sample per absolute
weather cell (station, seed, cell index; bounded uniform fade and Bernoulli outage). It consumes
no mutable RNG sequence and is independent of observation count. Agent telemetry
contains current/past link evidence and weather sampled no later than now; no
future weather sequence is returned. Scenario seeds/configuration are not a secure
hidden-information boundary against an adversarial agent with filesystem access.
This is synthetic stress weather, **not** ITU-R P.618 rain availability or a measured
atmospheric time series. Outage, band mismatch or failed margin prevents TX.
Forecast windows remain geometric/booking opportunities, not promises of RF rate,
energy admission or future successful delivery.

## Persistent ground terminals and finite capacity

Each ground terminal holds an actor, lock-ready time and last antenna direction.
First acquisition pays configured acquisition seconds. Handover additionally pays
spherical angular separation divided by configured slew rate; azimuth wrap and
zenith are handled geometrically. Cancellation, loss of eligibility or outage
releases lock. Reacquisition pays again; completion of one product does not grant
the next spacecraft an instantaneous usable antenna.

Acquisition does not pretend that payload data is being transmitted: spacecraft
TX seconds/energy stay zero until lock. Ground setup/tracking terminal-seconds and
watts×seconds are recorded separately. Initial antenna direction is unspecified;
first acquisition is the declared setup delay. Once acquired, ground tracking is
ideal: moving-target interception, tracking-rate/acceleration/servo errors, beacon
TX, acquisition waveform energy and ground battery/grid constraints remain open.

Per-terminal receiver bandwidth is distinct from station aggregate bit-rate and
aggregate bandwidth pools. `independent` policy greedily assigns remaining declared
pool capacity in stable terminal order; `equal-share` splits the bandwidth pool
between occupied terminals, then selects feasible modes. Actual aggregate rate and
bandwidth cannot exceed configured pools. This abstraction assumes isolated
receiving channels: it is not overlapping same-frequency FDM, interference, spectral
mask, TDMA slots or an optimal/fair network scheduler. Unused mode capacity may stay
idle; no fictitious rate is created by filling every last Hz.

## Booking imports and concurrency control

Local `terminal_booking_added/cancelled` and `external_reservations_imported` use
one booking conflict sweep. One record occupies one terminal and one actor radio
on a half-open interval. Import payload contains 1–1000 records with provider,
external ID, revision, expected previous revision, confirmed/cancelled status,
actor, station, interval and source. Provider/ID keys are collision-resistant.

Identical reimports are allocation-idempotent; stale revisions and conflicting
local/external allocations are rejected. A replacement revision truncates the old
allocation at now and creates a new immutable booking; history is preserved.
No backdating, unknown resources or allocations beyond the mission horizon.
Failed batches leave registry, local bookings, resources and event history intact.
Cancellation requires an existing live imported allocation. Provider records are
local evidence: this does not contact a provider or assert its real reservation.
Runtime revision/idempotency, event provenance and prefix replay remain mandatory.

## Serialization, ACK and exactly-once product credit

A native receipt records actual serialized bits and TX start/end. RF receipts also
record `ack_due_sec = serialization_end + 2*sampled_range/c + ack_processing_sec`.
This is an explicit ideal reliable return channel; ACK carries no modeled resource
contention, weather, loss or retransmission. Source bytes are freed as serialized;
there is no retained custody copy or automatic retry after source deletion.

Complete-product ACK boundaries advance even with no remaining radio command.
Native on-time score and managed artifact/request completion wait until the
latest required ACK. A product serialized before its deadline but acknowledged
after it is late. Repeated waits/replays do not create more bits or duplicate
credit. Request deadlines, TTL/failure and terminal states remain enforced.
Partial receipts are not complete delivered products; balance ledgers distinguish
serialized bits from acknowledged complete-product bytes. Agent ACK/reward is
causal; the ACK due timestamp in control-plane evidence is conditional on this
ideal ACK model, not a forecast of real future ground weather.

## Evidence and open B05 gates

42 new parametrized cases cover independent Friis, noise density and information
energy, inverse-square losses, coding/rolloff/overhead, discrete modes, full-band
filtering, body pointing, Doppler, numeric/schema rejection, visible-but-closed RF,
setup and slew, aggregate policies, seeded weather, lock loss/reacquisition,
read-only selection, DC/heat bounds, atomic budget failure and alias restoration,
local/imported conflicts, revisions, duplicate imports and portable event history,
late ACK, no early customer credit and mid-acquisition/transmission cold replay.

`python examples/network_mission/demo.py OUTPUT nominal|outage|blocked` builds
real runtime scenarios from capture through managed CPU and thermal-coupled RF.
Nominal/outage fulfill 2/2 requests; blocked RF correctly fulfills 0/2, without TX
or ground acquisition energy. Prefix and mid-transmission checkpoints replay.
SDK/installed CLI/full regression evidence lives in `docs/acceptance/B05_NETWORK.json`.

Delivered B05-A: B05.01–05 and B05.09 within this bounded engineering contract.
B05.06–08 and mixed B05.10 acceptance are now covered by the separate B05-B package: concurrent
ISL visibility/pointing, endpoint radio/CPU/energy admission, in-flight receiver
buffer reservation, delayed fragment availability, relay ownership/ledgers,
TTL/retry/loss/deduplication/custody and final exactly-once ACK. B05-B uses this runtime and lifecycle; do not infer it from the ground-only
B05-A profile or sequential legacy relay.
External RF/weather/antenna calibration and B09 fleet performance also remain open.
