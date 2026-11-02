"""Declared sampled RF engineering budgets; no waveform or calibrated atmosphere."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from space_env.runtime.schema import StrictModel

C_M_S = 299792458.0
BOLTZMANN_J_K = 1.380649e-23


class ModemMode(StrictModel):
    id: str = Field(min_length=1)
    information_rate_bps: float = Field(gt=0, le=1e12, strict=True)
    bits_per_symbol: int = Field(ge=1, le=12, strict=True)
    code_rate: float = Field(gt=0, le=1, strict=True)
    payload_fraction: float = Field(default=1., gt=0, le=1, strict=True)
    rolloff: float = Field(default=.25, ge=0, le=1, strict=True)
    required_eb_n0_db: float = Field(ge=-1.59, le=100, strict=True)
    implementation_loss_db: float = Field(default=0., ge=0, le=100, strict=True)

    @property
    def bandwidth_hz(self):
        return self.information_rate_bps * (1 + self.rolloff) / (
            self.bits_per_symbol * self.code_rate * self.payload_fraction)


class RFTransmitter(StrictModel):
    frequency_hz: float = Field(gt=0, le=1e12, strict=True)
    rf_power_w: float = Field(gt=0, le=1e6, strict=True)
    gain_dbi: float = Field(ge=-30, le=100, strict=True)
    feeder_loss_db: float = Field(default=0., ge=0, le=100, strict=True)
    pointing_loss_db: float = Field(default=0., ge=0, le=100, strict=True)
    # Omitted HPBW means declared ideal pointing, not inferred actual tracking.
    beamwidth_deg: float | None = Field(default=None, gt=0, le=180, strict=True)
    max_pointing_loss_db: float = Field(default=60., ge=0, le=200, strict=True)
    modes: tuple[ModemMode, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def unique_modes(self):
        if len({m.id for m in self.modes}) != len(self.modes):
            raise ValueError("duplicate modem mode ID")
        object.__setattr__(self, 'modes', tuple(sorted(self.modes, key=lambda m: (m.information_rate_bps, m.id))))
        return self


class RFReceiver(StrictModel):
    gt_db_k: float = Field(ge=-100, le=100, strict=True)
    min_frequency_hz: float = Field(gt=0, le=1e12, strict=True)
    max_frequency_hz: float = Field(gt=0, le=1e12, strict=True)
    bandwidth_hz: float = Field(gt=0, le=1e12, strict=True)
    doppler_tracking_limit_hz: float = Field(default=1e6, ge=0, le=1e9, strict=True)
    polarization_loss_db: float = Field(default=0., ge=0, le=100, strict=True)
    other_loss_db: float = Field(default=0., ge=0, le=200, strict=True)
    required_margin_db: float = Field(default=0., ge=0, le=100, strict=True)
    ack_processing_sec: float = Field(default=0., ge=0, le=3600, strict=True)
    acquisition_sec: float = Field(default=0., ge=0, le=3600, strict=True)
    slew_rate_deg_s: float = Field(default=10., gt=0, le=1000, strict=True)
    setup_power_w: float = Field(default=0., ge=0, le=1e9, strict=True)
    tracking_power_w: float = Field(default=0., ge=0, le=1e9, strict=True)
    aggregate_rate_bps: float | None = Field(default=None, gt=0, le=1e13, strict=True)
    aggregate_bandwidth_hz: float | None = Field(default=None, gt=0, le=1e13, strict=True)
    bandwidth_policy: Literal['independent', 'equal-share'] = 'independent'

    @model_validator(mode="after")
    def ordered_band(self):
        if self.max_frequency_hz <= self.min_frequency_hz:
            raise ValueError('receiver frequency band must be ordered')
        if self.bandwidth_policy == 'equal-share' and self.aggregate_bandwidth_hz is None:
            raise ValueError('equal-share requires aggregate bandwidth')
        return self


class RadioWeather(StrictModel):
    source: str = Field(min_length=1)
    seed: int = Field(default=0, ge=0, le=2**63-1, strict=True)
    step_sec: float = Field(default=10., ge=.01, le=3600, strict=True)
    max_fade_db: float = Field(default=0., ge=0, le=200, strict=True)
    outage_probability: float = Field(default=0., ge=0, le=1, strict=True)


class RadioNetworkSettings(StrictModel):
    model: Literal['sampled-rf-ground/1'] = 'sampled-rf-ground/1'
    source: str = Field(min_length=1)
    parameter_status: Literal['synthetic', 'published', 'calibrated']
    sample_sec: float = Field(default=1., ge=.01, le=30, strict=True)
    transmitters: dict[str, RFTransmitter] = Field(min_length=1, max_length=64)
    receivers: dict[str, RFReceiver] = Field(min_length=1, max_length=128)
    weather: dict[str, RadioWeather] = Field(default_factory=dict)
    max_intervals: int = Field(default=100000, ge=1, le=1000000, strict=True)


def rf_budget(tx: RFTransmitter, rx: RFReceiver, range_m: float, *,
              fade_db: float = 0., pointing_error_deg: float = 0., doppler_hz: float = 0.,
              bandwidth_hz: float | None = None, rate_cap_bps: float | None = None):
    """Net information-bit Eb/N0; occupied bandwidth accounts for code/overhead.

    Carrier power is held fixed across modes. G/T includes receiver feeder/noise
    effects, so they must not be charged again as separate receiver losses.
    """
    for value in (range_m, fade_db, pointing_error_deg, doppler_hz):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('RF inputs must be finite numbers')
    if range_m <= 0 or fade_db < 0 or not 0 <= pointing_error_deg <= 180:
        raise ValueError('invalid RF range/fade/pointing')
    for value in (bandwidth_hz, rate_cap_bps):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError('RF allocation must be finite and nonnegative')
    fspl = 20 * math.log10(4 * math.pi * range_m * tx.frequency_hz / C_M_S)
    pattern = (min(tx.max_pointing_loss_db, 12 * (pointing_error_deg / tx.beamwidth_deg)**2)
               if tx.beamwidth_deg is not None else 0.)
    eirp = 10 * math.log10(tx.rf_power_w) + tx.gain_dbi - tx.feeder_loss_db - tx.pointing_loss_db - pattern
    cn0 = eirp - fspl + rx.gt_db_k - rx.polarization_loss_db - rx.other_loss_db - fade_db - 10*math.log10(BOLTZMANN_J_K)
    bandwidth = min(rx.bandwidth_hz, bandwidth_hz) if bandwidth_hz is not None else rx.bandwidth_hz
    candidates = []
    for m in tx.modes:
        margin = cn0 - 10*math.log10(m.information_rate_bps) - m.required_eb_n0_db - m.implementation_loss_db
        band_ok = (tx.frequency_hz+doppler_hz-m.bandwidth_hz/2 >= rx.min_frequency_hz
                   and tx.frequency_hz+doppler_hz+m.bandwidth_hz/2 <= rx.max_frequency_hz
                   and abs(doppler_hz) <= rx.doppler_tracking_limit_hz)
        eligible = (band_ok and m.bandwidth_hz <= bandwidth and margin >= rx.required_margin_db
                    and (rate_cap_bps is None or m.information_rate_bps <= rate_cap_bps))
        candidates.append({'id':m.id, 'information_rate_bps':m.information_rate_bps,
                           'occupied_bandwidth_hz':m.bandwidth_hz, 'margin_db':margin,
                           'band_compatible':band_ok, 'eligible':eligible})
    selected = next((m for m in reversed(candidates) if m['eligible']), None)
    return {'fspl_db':fspl, 'eirp_dbw':eirp, 'cn0_db_hz':cn0, 'fade_db':fade_db,
            'pointing_error_deg':pointing_error_deg, 'pattern_loss_db':pattern, 'doppler_hz':doppler_hz,
            'selected_mode':None if selected is None else selected['id'],
            'rate_bps':0. if selected is None else selected['information_rate_bps'],
            'occupied_bandwidth_hz':0. if selected is None else selected['occupied_bandwidth_hz'],
            'modes':candidates}
