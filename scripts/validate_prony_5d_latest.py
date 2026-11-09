"""Controlled 5D exponential-moment pilot for the updated NumPy twin.

Writes results_5d_latest.json under FRI_RUN_DIR (default: runs/).
"""

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("AUREIS_TWIN_DIR", ROOT.parent / "aureis-1d-twin-latest")).resolve()
RUN_DIR = Path(os.environ.get("FRI_RUN_DIR", ROOT / "runs")).resolve()
if not (REPO / "experiments" / "5d_calorimetry" / "experiment.py").is_file():
    raise SystemExit(f"5D simulator checkout missing at {REPO}. Set AUREIS_TWIN_DIR (see README).")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from run import build_readout, load_experiment

NS = 1e-9
PERIOD = 25 * NS
BINS = np.arange(1, 17)
FREQ = BINS / PERIOD
RIDGE = 1e-3


def target(times, amplitudes=None):
    exponentials = np.exp(-2j * np.pi * FREQ[:, None] * np.asarray(times))
    return exponentials if amplitudes is None else exponentials @ amplitudes


def fit_weights(samples, targets, relative_ridge):
    u, s, vh = np.linalg.svd(samples, full_matrices=False)
    damping = s / (s**2 + (relative_ridge * s[0])**2)
    return ((targets @ vh.conj().T) * damping) @ u.conj().T


def scores(moments, times, amplitudes):
    ideal = target(times, amplitudes)
    hankel = np.lib.stride_tricks.sliding_window_view(moments, 9)
    u, singular_values, _ = np.linalg.svd(hankel, full_matrices=False)
    subspace = u[:, :2]
    shift = np.linalg.lstsq(subspace[:-1], subspace[1:], rcond=None)[0]
    poles = np.linalg.eigvals(shift)
    arrival = np.sort(np.mod(-np.angle(poles) * PERIOD / (2 * np.pi), PERIOD))
    amplitudes_fit = np.linalg.lstsq(target(arrival), moments, rcond=None)[0]
    return {
        "moment_relative": float(np.linalg.norm(moments - ideal) / np.linalg.norm(ideal)),
        "hankel_sigma3_over_sigma2": float(singular_values[2] / singular_values[1]),
        "max_time_error_ps": float(np.max(np.abs(arrival - times)) * 1e12),
        "amplitude_relative": float(
            np.linalg.norm(amplitudes_fit - amplitudes) / np.linalg.norm(amplitudes)
        ),
    }


def main():
    exp = load_experiment("5d_calorimetry")
    t_in, sipm, chain = build_readout(exp)
    t_nom = chain.sample_times()
    settings = chain.params()
    noise_groups = {
        "sipm": [(sipm, "noise_psd"), (sipm, "ou_drift_std")],
        "tia": [(chain.tia, "noise_psd_onesided"), (chain.tia, "temperature")],
        "jitter": [(chain.memory, "jitter_rms")],
        "memory_thermal": [(chain.memory, "temperature")],
        "adc_noise": [(chain.adc, "noise_rms")],
    }
    noise_defaults = {
        name: [(obj, attr, getattr(obj, attr)) for obj, attr in settings_group]
        for name, settings_group in noise_groups.items()
    }

    def forward(times, amplitudes, seed, enabled=()):
        for name, settings_group in noise_defaults.items():
            for obj, attr, default in settings_group:
                setattr(obj, attr, default if name in enabled else 0.0)
        rng = np.random.default_rng(seed)
        sipm.rng = rng
        _, current = sipm.simulate(times, amps=amplitudes)
        result = chain.simulate(current, rng)
        result["tia_nominal"] = np.interp(t_nom, t_in, result["tia_v"])
        result["adc_v"] = chain.adc.to_volts(result["adc_out"]) - chain.adc.baseline
        return result

    def single_columns(times, tap):
        return np.column_stack([forward([t], [1.0], 0)[tap] for t in times])

    cal_times = np.linspace(2, 20, 361) * NS
    rng = np.random.default_rng(20261107)
    # Stratified times with independent fractional positions on the 5 ps grid.
    inside = (2 + (np.arange(360) + rng.uniform(0.1, 0.9, 360)) * 0.05) * NS
    late = np.linspace(20.025, 24.925, 99) * NS
    weights = {}
    single = []
    holdout_memory = None
    for tap in ("tia_nominal", "memory_v"):
        train = single_columns(cal_times, tap)
        hold = single_columns(inside, tap)
        edge = single_columns(late, tap)
        if tap == "memory_v":
            holdout_memory = np.concatenate((hold, edge), axis=1)
        for relative_ridge in (1e-2, RIDGE, 1e-4):
            c = fit_weights(train, target(cal_times), relative_ridge)
            if relative_ridge == RIDGE:
                weights[tap] = c
            for name, observed, times in (("inside", hold, inside), ("late", edge, late)):
                error = c @ observed - target(times)
                single.append({
                    "tap": tap, "ridge": relative_ridge, "interval": name,
                    "rms": float(np.sqrt(np.mean(np.abs(error)**2))),
                    "p95_per_arrival_rms": float(
                        np.percentile(np.sqrt(np.mean(np.abs(error)**2, axis=0)), 95)
                    ),
                    "peak_moment_error": float(np.abs(error).max()),
                    "max_weight_row_norm_pe_per_v": float(np.linalg.norm(c, axis=1).max()),
                })

    c_mem = weights["memory_v"]
    wide_times = np.linspace(2, 24.95, 460) * NS
    c_wide = fit_weights(single_columns(wide_times, "memory_v"), target(wide_times), RIDGE)
    wide_error = c_wide @ holdout_memory - target(np.concatenate((inside, late)))
    wide_window = {
        "calibration_arrival_window_ns": [2, 24.95],
        "inside_rms": float(np.sqrt(np.mean(np.abs(wide_error[:, :inside.size])**2))),
        "late_rms": float(np.sqrt(np.mean(np.abs(wide_error[:, inside.size:])**2))),
        "max_weight_row_norm_pe_per_v": float(np.linalg.norm(c_wide, axis=1).max()),
    }
    # Baseline comparison: truncated Fourier quadrature and one reference pulse.
    quadrature = np.full(t_nom.size, 1.0 / chain.memory.samp_rate)
    quadrature[[0, -1]] *= 0.5
    fourier = target(t_nom) * quadrature
    template_spectrum = fourier @ forward([0.0], [1.0], 0)["memory_v"]
    n_seeds = 64
    trials = []
    for separation_ns in (0.3, 1.3, 3.0):
        times = np.array([6.037, 6.037 + separation_ns]) * NS
        for amplitude in (1.0, 5.0):
            amplitudes = np.full(2, amplitude)
            ideal = target(times, amplitudes)
            clean = forward(times, amplitudes, 0)
            clean_mem = c_mem @ clean["memory_v"]
            clean_adc = c_mem @ clean["adc_v"]
            measurements = {
                "oracle": [ideal],
                "tia_nominal_clean": [weights["tia_nominal"] @ clean["tia_nominal"]],
                "memory_fourier_division_clean": [(fourier @ clean["memory_v"]) / template_spectrum],
                "memory_calibrated_clean": [clean_mem],
                "adc_quantization_clean": [clean_adc],
            }
            increments = {}
            for name in (*noise_groups, "all"):
                enabled = tuple(noise_groups) if name == "all" else (name,)
                tap = "adc_v" if name in ("adc_noise", "all") else "memory_v"
                reference = clean_adc if tap == "adc_v" else clean_mem
                output = [forward(times, amplitudes, seed, enabled) for seed in range(n_seeds)]
                key = f"{tap}_{name}"
                measurements[key] = [c_mem @ result[tap] for result in output]
                increments[key] = [
                    float(np.linalg.norm(z - reference) / np.linalg.norm(ideal))
                    for z in measurements[key]
                ]
                if name == "all":
                    measurements["memory_v_all"] = [c_mem @ result["memory_v"] for result in output]
                    increments["adc_minus_memory_all"] = [
                        float(np.linalg.norm(c_mem @ (result["adc_v"] - result["memory_v"]))
                              / np.linalg.norm(ideal)) for result in output
                    ]
            adc_input = clean["memory_v"] + chain.adc.baseline
            for name, measured in measurements.items():
                stats = [scores(z, times, amplitudes) for z in measured]
                trials.append({
                    "separation_ns": separation_ns, "amplitude_each_pe": amplitude,
                    "method": name, "n": len(stats),
                    **{f"median_{key}": float(np.median([s[key] for s in stats]))
                       for key in stats[0]},
                    "p95_max_time_error_ps": float(
                        np.percentile([s["max_time_error_ps"] for s in stats], 95)
                    ),
                    "both_within_100ps": float(np.mean([s["max_time_error_ps"] <= 100 for s in stats])),
                    "median_increment_relative": (
                        float(np.median(increments[name])) if name in increments else None
                    ),
                    "median_adc_minus_memory_all_relative": (
                        float(np.median(increments["adc_minus_memory_all"]))
                        if name == "adc_v_all" else None
                    ),
                    "clean_adc_outside_range_fraction": float(np.mean(
                        (adc_input < chain.adc.v_low) | (adc_input >= chain.adc.v_high)
                    )),
                })
    # The existing 5D data use the upstream placeholder photon generator.
    # Use detected photons to score the estimate only, with one p.e. per detection.
    saved_event_check = None
    path = ROOT / "data" / "synthetic_5d" / "5d_latest_100events.npz"
    if path.is_file():
        with np.load(path) as data:
            np.testing.assert_allclose(data["sample_times"], t_nom, atol=1e-18, rtol=0)
            adc = chain.adc.to_volts(data["adc_out"]) - chain.adc.baseline
            errors = {"all_nonempty": [], "within_calibration": [], "outside_calibration": []}
            counts = []
            for event_id, samples in zip(data["event_id"], adc):
                detected = data["detected_t"][data["detected_event"] == event_id]
                counts.append(detected.size)
                if not detected.size:
                    continue
                ideal = target(detected, np.ones(detected.size))
                relative_error = float(np.linalg.norm(c_mem @ samples - ideal) / np.linalg.norm(ideal))
                errors["all_nonempty"].append(relative_error)
                interval = (
                    "within_calibration" if np.all((detected >= cal_times[0]) & (detected <= cal_times[-1]))
                    else "outside_calibration"
                )
                errors[interval].append(relative_error)
            saved_event_check = {
                "file": path.relative_to(ROOT).as_posix(),
                "events": len(counts),
                "empty_events": counts.count(0),
                "max_detected_count": max(counts),
                "more_than_8_detected": sum(k > 8 for k in counts),
                "relative_moment_error": {
                    name: {"n": len(values), "median": float(np.median(values)),
                           "p95": float(np.percentile(values, 95))}
                    for name, values in errors.items() if values
                },
            }
    payload = {
        "simulator_checkout": REPO.name,
        "experiment": exp.SIPM,
        "hardware": settings,
        "noise_defaults": {name: {attr: value for _, attr, value in group}
                           for name, group in noise_defaults.items()},
        "calibration_arrival_window_ns": [2, 20],
        "training_step_ps": 50,
        "period_ns": PERIOD / NS,
        "bins": BINS.tolist(),
        "fixed_ridge": RIDGE,
        "calibration_holdout": single,
        "wide_window_diagnostic": wide_window,
        "two_pulse_trials": trials,
        "saved_5d_events": saved_event_check,
        "scope": "Known K=2, fixed pair position, default correlated SiPM effects off.",
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "results_5d_latest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
