"""Offline Q5-B oracle: closed-form gyrostat, independent RK4 and contact roots.

No space_env or scipy imports. ERFA is used ONLY for explicitly shared Earth
orientation; no claim of independent frame/EOP truth is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import erfa

MU = 3.986004418e14
RE = 6378137.0
RP = 6356752.314245
G0 = 9.80665
C = 299792458.0


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    return [x / norm(a) for x in a]


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(row) for row in zip(*a)]


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def rodrigues(axis, angle):
    x, y, z = unit(axis)
    c, s, d = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [
        [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
        [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
        [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
    ]


def gyro_inputs(torque, momentum):
    axis, angle = unit([0.3, 0.4, 0.5]), 0.8
    return {
        "inertia_kg_m2": [2.0, 2.0, 3.0],
        "wheel_inertia_kg_m2": 0.01,
        "initial_quaternion_wxyz": [math.cos(angle / 2)]
        + [v * math.sin(angle / 2) for v in axis],
        "initial_rotation": rodrigues(axis, angle),
        "omega_body_rad_s": [0.12, -0.08, 0.23],
        "h_z_nms": momentum,
        "motor_z_nm": torque,
        "momentum_limit_nms": 0.13,
        "torque_limit_nm": 0.02,
        "step_sec": 0.01,
        "boresight_body": [0.6, 0.0, 0.8],
        "target_gcrs": unit([1.0, -2.0, 0.5]),
    }


def gyro(p, t):
    """Exact symmetric gyrostat: constant inertial H, piecewise-linear wheel h."""
    ip, _, iz = p["inertia_kg_m2"]
    w, h0, torque = p["omega_body_rad_s"], p["h_z_nms"], p["motor_z_nm"]
    h3 = iz * w[2] + h0
    h_body0 = [ip * w[0], ip * w[1], h3]
    h_inertial = mv(p["initial_rotation"], h_body0)
    cap_time = (
        math.inf
        if torque == 0
        else (math.copysign(p["momentum_limit_nms"], torque) - h0) / torque
    )
    active = min(t, cap_time)
    h = h0 + torque * active
    integrated_h = h0 * t + torque * (active * active / 2 + active * (t - active))
    phi = h3 * t / iz - integrated_h / iz - h3 * t / ip
    rotation = mm(
        mm(rodrigues(h_inertial, norm(h_inertial) * t / ip), p["initial_rotation"]),
        rodrigues([0, 0, 1], phi),
    )
    momentum = mv(transpose(rotation), h_inertial)
    omega = [momentum[0] / ip, momentum[1] / ip, (momentum[2] - h) / iz]
    boresight = mv(rotation, p["boresight_body"])
    cross = [
        boresight[1] * p["target_gcrs"][2] - boresight[2] * p["target_gcrs"][1],
        boresight[2] * p["target_gcrs"][0] - boresight[0] * p["target_gcrs"][2],
        boresight[0] * p["target_gcrs"][1] - boresight[1] * p["target_gcrs"][0],
    ]
    pointing = math.degrees(math.atan2(norm(cross), dot(boresight, p["target_gcrs"])))
    return rotation, omega, [0.0, 0.0, h], pointing, max(0.0, t - cap_time)


def orbit(p, t):
    n = math.sqrt(MU / p["radius_m"] ** 3)
    angle = p["phase_rad"] + n * t
    r = [p["radius_m"] * math.cos(angle), p["radius_m"] * math.sin(angle), 0.0]
    v = [-p["radius_m"] * n * math.sin(angle), p["radius_m"] * n * math.cos(angle), 0.0]
    rot = mm(
        rodrigues([0, 0, 1], p["raan_rad"]), rodrigues([1, 0, 0], p["inclination_rad"])
    )
    return mv(rot, r) + mv(rot, v)


def earth_rotation(t):
    # Constant synthetic EOP on 2026-07-18..19; no leap transition in this scope.
    u1, u2 = erfa.dtf2d("UTC", 2026, 7, 18, 12, 0, 0.0)
    utc = (float(u1), float(u2) + t / 86400.0)
    tt = erfa.taitt(*erfa.utctai(*utc))
    ut1 = erfa.utcut1(*utc, 0.1)
    arcsec = math.pi / (180 * 3600)
    return erfa.c2t06a(*tt, *ut1, 0.05 * arcsec, 0.25 * arcsec).tolist()


def ground(p):
    lat, lon = math.radians(p["lat_deg"]), math.radians(p["lon_deg"])
    ecc2 = 1 - (RP / RE) ** 2
    normal = RE / math.sqrt(1 - ecc2 * math.sin(lat) ** 2)
    up = [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
    site = [
        (normal + p["alt_m"]) * up[0],
        (normal + p["alt_m"]) * up[1],
        (normal * (1 - ecc2) + p["alt_m"]) * up[2],
    ]
    return site, up, lat, lon


def look(p, site, t):
    position = orbit(p, t)[:3]
    rot = earth_rotation(t)
    fixed = mv(rot, position)
    ground_fixed, up, lat, lon = ground(site)
    line = [x - y for x, y in zip(fixed, ground_fixed)]
    distance = norm(line)
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, dot(line, up) / distance))))
    east = -math.sin(lon) * line[0] + math.cos(lon) * line[1]
    north = (
        -math.sin(lat) * math.cos(lon) * line[0]
        - math.sin(lat) * math.sin(lon) * line[1]
        + math.cos(lat) * line[2]
    )
    azimuth = math.degrees(math.atan2(east, north)) % 360
    target = mv(transpose(rot), ground_fixed)
    direction = unit([x - y for x, y in zip(target, position)])
    return distance, elevation, azimuth, direction


def windows(p, site):
    # Independent 5s scan and bisection, not production Brent/access functions.
    duration, step = p["duration_sec"], 5.0
    previous = look(p, site, 0.0)[1] - site["min_elevation_deg"]
    begin, result = (0.0 if previous >= 0 else None), []
    for i in range(1, int(duration / step) + 1):
        left, right = (i - 1) * step, i * step
        after = look(p, site, right)[1] - site["min_elevation_deg"]
        if (previous >= 0) != (after >= 0):
            lo, hi, sign = left, right, previous >= 0
            while hi - lo > 1e-7:
                mid = (lo + hi) / 2
                if (look(p, site, mid)[1] >= site["min_elevation_deg"]) == sign:
                    lo = mid
                else:
                    hi = mid
            root = (lo + hi) / 2
            if after >= 0:
                begin = root
            elif begin is not None:
                result.append([begin, root])
                begin = None
        previous = after
    if begin is not None:
        result.append([begin, duration])
    return result


def cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def rk4(rhs, t, state, step):
    k1 = rhs(t, state)
    k2 = rhs(t + step / 2, [y + step * x / 2 for y, x in zip(state, k1)])
    k3 = rhs(t + step / 2, [y + step * x / 2 for y, x in zip(state, k2)])
    k4 = rhs(t + step, [y + step * x for y, x in zip(state, k3)])
    return [
        y + step * (a + 2 * b + 2 * c + d) / 6
        for y, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]


def orientation_initial(p):
    return (
        [x for row in p["initial_rotation"] for x in row]
        + p["omega_body_rad_s"]
        + [0.0, 0.0, p["h_z_nms"]]
    )


def orientation_rhs(p, state, external):
    # Matrix kinematics, NOT quaternion derivatives, and component Euler balance.
    rotation = [state[i : i + 3] for i in (0, 3, 6)]
    wx, wy, wz = state[9:12]
    skew = [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]]
    derivative = mm(rotation, skew)
    momentum = [
        i * w + h for i, w, h in zip(p["inertia_kg_m2"], state[9:12], state[12:15])
    ]
    gyro_torque = cross(state[9:12], momentum)
    motor = [0.0, 0.0, p["motor_z_nm"]]
    acceleration = [
        (e - u - g) / i
        for e, u, g, i in zip(external, motor, gyro_torque, p["inertia_kg_m2"])
    ]
    return [x for row in derivative for x in row] + acceleration + motor


def orientation_at(p, t, external, step):
    state = orientation_initial(p)
    count = math.ceil(t / step)
    delta = t / count
    for i in range(count):
        state = rk4(
            lambda _clock, y: orientation_rhs(p, y, external), i * delta, state, delta
        )
    return state


def burn(p):
    """Independent Cartesian/matrix RK4, analytic orientation for centered case."""
    force, isp, mass = p["thrust_n"], p["isp_s"], p["mass_kg"]
    attitude = p["attitude"]
    external = cross(
        p["lever_arm_body_m"], [force * x for x in attitude["boresight_body"]]
    )
    offset = any(external)

    def rotation_at(t):
        if not offset:
            return gyro(attitude, t)[0]
        state = orientation_at(attitude, t, external, p["oracle_step_sec"])
        return [state[i : i + 3] for i in (0, 3, 6)]

    def rhs(t, state):
        position, velocity = state[-6:-3], state[-3:]
        rotation = [state[i : i + 3] for i in (0, 3, 6)] if offset else rotation_at(t)
        direction = mv(rotation, attitude["boresight_body"])
        radius = norm(position)
        current = mass - force / (isp * G0) * t
        cartesian = velocity + [
            -MU * x / radius**3 + force / current * d
            for x, d in zip(position, direction)
        ]
        return (
            orientation_rhs(attitude, state[:15], external) + cartesian
            if offset
            else cartesian
        )

    state = (orientation_initial(attitude) if offset else []) + list(p["initial_si"])
    result, samples, directions = [], [], []
    step = p["oracle_step_sec"]
    count = round(p["duration_sec"] / step)
    selected = {round(t / step) for t in p["times_sec"]}
    directions.append(mv(attitude["initial_rotation"], attitude["boresight_body"]))
    for i in range(count):
        state = rk4(rhs, i * step, state, step)
        t = (i + 1) * step
        if offset:
            rotation = [state[j : j + 3] for j in (0, 3, 6)]
            orient = state[:15]
        else:
            rotation, w, h, _, _ = gyro(attitude, t)
            orient = [x for row in rotation for x in row] + w + h
        directions.append(mv(rotation, attitude["boresight_body"]))
        if i + 1 in selected:
            result.append(state[-6:] + [mass - force / (isp * G0) * t])
            samples.append(orient)
    total = [0.0] * 3
    for i, direction in enumerate(directions):
        weight = 1 if i in (0, count) else (4 if i % 2 else 2)
        total = [a + weight * force * d * step / 3 for a, d in zip(total, direction)]
    return {
        "states_si_mass": result,
        "attitude_samples": samples,
        "impulse_ns": total,
        "directions": [
            mv(rotation_at(t), attitude["boresight_body"])
            for t in p["direction_times_sec"]
        ],
    }


def build():
    cases = []
    times = [0.01, 1.0, 7.0, 7.06, 10.0, 20.0]
    for name, torque, h in (
        ("offaxis-coast", 0.0, 0.03),
        ("offaxis-wheel", 0.002, 0.03),
        ("offaxis-positive-cap", 0.017, 0.01),
        ("offaxis-negative-cap", -0.017, -0.01),
        ("offaxis-reverse-through-zero", -0.017, 0.1),
    ):
        p = gyro_inputs(torque, h)
        expected = []
        for t in times:
            rot, w, momentum, pointing, saturated = gyro(p, t)
            expected.append(
                [x for row in rot for x in row] + w + momentum + [pointing, saturated]
            )
        cases.append(
            {
                "id": name,
                "kind": "attitude",
                "inputs": p,
                "times_sec": times,
                "expected": expected,
            }
        )
    forced = gyro_inputs(-0.017, 0.1)
    forced["inertia_kg_m2"] = [2.0, 3.0, 4.0]
    forced["external_body_nm"] = [0.0, 0.0, -0.017]
    forced_times = [0.01, 1.0, 5.0, 10.0]
    values = []
    for t in forced_times:
        state = orientation_at(forced, t, forced["external_body_nm"], 0.0025)
        matrix = [state[i : i + 3] for i in (0, 3, 6)]
        bore = mv(matrix, forced["boresight_body"])
        error = math.degrees(
            math.atan2(
                norm(cross(bore, forced["target_gcrs"])),
                dot(bore, forced["target_gcrs"]),
            )
        )
        values.append(state + [error, 0.0])
    cases.append(
        {
            "id": "offaxis-balanced-unloading",
            "kind": "forced_attitude",
            "inputs": forced,
            "times_sec": forced_times,
            "expected": values,
        }
    )
    attitude = gyro_inputs(0.0, 0.0)
    p = {
        "attitude": attitude,
        "lever_arm_body_m": [0.0, 0.0, 0.0],
        "initial_si": [7e6, 0.0, 0.0, 0.0, math.sqrt(MU / 7e6), 0.0],
        "thrust_n": 0.5,
        "isp_s": 300.0,
        "mass_kg": 100.0,
        "dry_mass_kg": 90.0,
        "mu_m3_s2": MU,
        "g0_m_s2": G0,
        "duration_sec": 10.0,
        "times_sec": [1.0, 5.0, 10.0],
        "oracle_step_sec": 0.005,
        "direction_times_sec": [0.0037, 0.127, 0.333, 1.0, 3.7, 9.999],
        "epoch_utc": "2026-07-18T12:00:00Z",
    }
    cases.append(
        {
            "id": "offaxis-rotating-burn",
            "kind": "burn",
            "inputs": p,
            "times_sec": p["times_sec"],
            "expected": burn(p),
        }
    )
    offset = dict(p)
    offset["attitude"] = gyro_inputs(0.0, 0.01)
    offset["attitude"]["inertia_kg_m2"] = [2.0, 3.0, 4.0]
    offset["lever_arm_body_m"] = [0.1, 0.2, -0.05]
    cases.append(
        {
            "id": "offaxis-offset-burn",
            "kind": "burn",
            "inputs": offset,
            "times_sec": offset["times_sec"],
            "expected": burn(offset),
        }
    )
    sites = [
        ("equatorial", 0.0, 0.0),
        ("iberia", 40.0, -3.0),
        ("australia", -35.0, 149.0),
        ("arctic", 78.0, 15.0),
    ]
    times = [0.0, 600.0, 1800.0, 3600.0, 6000.0, 9000.0, 12000.0]
    for plane, inc, raan in (("plane-a", 0.7, 0.2), ("plane-b", 1.7, 1.4)):
        for name, lat, lon in sites:
            p = {
                "radius_m": 7e6,
                "inclination_rad": inc,
                "raan_rad": raan,
                "phase_rad": 0.4,
                "duration_sec": 12000.0,
                "sample_sec": 5.0,
                "mu_m3_s2": MU,
                "epoch_utc": "2026-07-18T12:00:00Z",
                "dut1_s": 0.1,
                "xp_arcsec": 0.05,
                "yp_arcsec": 0.25,
                "earth_equator_m": RE,
                "earth_polar_m": RP,
                "c_m_s": C,
                "frequency_hz": 2.2e9,
                "station": {
                    "name": name,
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "alt_m": 100.0,
                    "min_elevation_deg": 5.0,
                },
                "initial_si": orbit(
                    {
                        "radius_m": 7e6,
                        "inclination_rad": inc,
                        "raan_rad": raan,
                        "phase_rad": 0.4,
                    },
                    0.0,
                ),
            }
            expected = []
            for t in times:
                distance, el, az, los = look(p, p["station"], t)
                rate = (
                    look(p, p["station"], t + 0.1)[0]
                    - look(p, p["station"], t - 0.1)[0]
                ) / 0.2
                expected.append(
                    [distance, rate, el, az, -p["frequency_hz"] * rate / C] + los
                )
            cases.append(
                {
                    "id": plane + "-" + name,
                    "kind": "contact",
                    "inputs": p,
                    "times_sec": times,
                    "expected": {
                        "looks": expected,
                        "windows": windows(p, p["station"]),
                    },
                }
            )
    body = {
        "schema": "space-env.pointing-vectors/1",
        "provenance": {
            "generator": "scripts/generate_pointing_vectors.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independence": "analytic-gyrostat+independent-rk4+independent-contact-equations/shared-erfa-inputs",
            "erfa_version": erfa.__version__,
            "shared_inputs": [
                "SI constants",
                "initial state and synthetic parameters",
                "ERFA c2t06a and leap table",
                "constant synthetic EOP",
            ],
            "scope": "frozen synthetic equation comparisons; no hardware or flight truth; no production imports",
        },
        "cases": cases,
    }
    return {**body, "sha256": digest(body)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(), indent=2, allow_nan=False) + "\n")
