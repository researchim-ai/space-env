"""
Опциональные отраслевые библиотеки. Тяжёлые пакеты импортируются только при вызове.
Их наличие не заменяет validation.reference_report и не переводит TEME в GCRF.
"""
from space_env.integrations.orekit_prop import propagate_keplerian_m
from space_env.integrations.sgp4_prop import orbit_from_tle
from space_env.integrations.spice_ephem import load_spice_kernels, sun_position_spice_m

__all__ = [
    "load_spice_kernels",
    "orbit_from_tle",
    "propagate_keplerian_m",
    "sun_position_spice_m",
]
