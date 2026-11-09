"""Precompute and reload fixed moment weights for a single detected 1 p.e. pulse.

Use the NumPy environment in aureis-1d-twin-latest.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from validate_prony_5d_latest import (
    BINS, NS, PERIOD, RIDGE, RUN_DIR, build_readout, fit_weights, load_experiment, target,
)


STAGES = ("tia_nominal", "adc_v")
REGIONS = {"inside": (2.0, 20.0), "late": (20.025, 24.925)}
CONDITIONS = ("no_added_noise", "default_noise")
DEFAULT_DIR = RUN_DIR / "single_pulse_5d"


def setup(noise):
    experiment = load_experiment("5d_calorimetry")
    t_in, sipm, chain = build_readout(experiment)
    hardware = chain.params()
    if not noise:
        sipm.noise_psd = sipm.ou_drift_std = 0.0
        chain.tia.noise_psd_onesided = chain.tia.temperature = 0.0
        chain.memory.temperature = chain.memory.jitter_rms = 0.0
        chain.adc.noise_rms = 0.0
    return (t_in, sipm, chain), hardware


def read_pulse(readout, arrival, rng):
    t_in, sipm, chain = readout
    sipm.rng = rng
    _, current = sipm.simulate([arrival], amps=[1.0])
    out = chain.simulate(current, rng)
    tia = np.interp(chain.sample_times(), t_in, out["tia_v"])
    adc = chain.adc.to_volts(out["adc_out"]) - chain.adc.baseline
    return np.stack((tia, adc))


def calibrate(out_dir):
    readout, hardware = setup(noise=False)
    train_times = np.linspace(2.0, 20.0, 361) * NS
    rng = np.random.default_rng(0)
    waveforms = np.stack([read_pulse(readout, arrival, rng) for arrival in train_times])
    ideal = target(train_times)
    weights = {
        stage: fit_weights(waveforms[:, index, :].T, ideal, RIDGE)
        for index, stage in enumerate(STAGES)
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "weights.npz"
    np.savez_compressed(
        path, **weights, train_times_s=train_times,
        sample_times_s=readout[2].sample_times(),
        bins=BINS, period_s=PERIOD, relative_ridge=RIDGE,
        hardware_json=json.dumps(hardware),
    )
    print(f"Saved fixed weights to {path}", flush=True)


def estimate_arrival(moments):
    # Rank-1 ESPRIT. Consecutive Hankel rows differ by the root z for K=1.
    hankel = np.lib.stride_tricks.sliding_window_view(moments, 9)
    left = np.linalg.svd(hankel, full_matrices=False)[0][:, 0]
    upper, lower = left[:-1], left[1:]
    denominator = np.vdot(upper, upper).real
    if not np.any(moments) or denominator == 0:
        return np.nan
    z = np.vdot(upper, lower) / denominator
    if z == 0:
        return np.nan
    return float(np.mod(-np.angle(z) * PERIOD / (2 * np.pi), PERIOD))


def summarize(estimates, truth, moments):
    error_ps = (estimates - truth) * 1e12
    valid = np.isfinite(error_ps)
    absolute = np.abs(error_ps[valid])
    moment_error = np.linalg.norm(moments - target(truth).T, axis=1) / np.sqrt(BINS.size)
    return {
        "trials": int(truth.size),
        "invalid_estimates": int(np.count_nonzero(~valid)),
        "median_moment_relative_error": float(np.median(moment_error)),
        "median_signed_error_ps": float(np.median(error_ps[valid])) if absolute.size else None,
        "median_absolute_error_ps": float(np.median(absolute)) if absolute.size else None,
        "p95_absolute_error_ps": float(np.percentile(absolute, 95)) if absolute.size else None,
        "within_100ps": float(np.mean(np.abs(error_ps) <= 100)),
    }


def evaluate(out_dir, n_trials, seed):
    quiet, hardware = setup(noise=False)
    noisy, _ = setup(noise=True)
    with np.load(out_dir / "weights.npz") as artifact:
        if (json.loads(artifact["hardware_json"].item()) != hardware
                or not np.array_equal(artifact["sample_times_s"], quiet[2].sample_times())
                or not np.array_equal(artifact["bins"], BINS)
                or artifact["period_s"].item() != PERIOD):
            raise ValueError("Saved weights do not match this readout or moment definition. Recalibrate.")
        weights = np.stack([artifact[stage] for stage in STAGES])

    arrival_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    arrival_rng = np.random.default_rng(arrival_seed)
    noise_rng = np.random.default_rng(noise_seed)
    regions = list(REGIONS)
    truth = np.stack([arrival_rng.uniform(lo, hi, n_trials) * NS for lo, hi in REGIONS.values()])
    trial_seeds = noise_rng.integers(0, 2**32, size=truth.shape)
    moments = np.empty((len(CONDITIONS), len(regions), n_trials, len(STAGES), BINS.size),
                       dtype=np.complex128)
    estimates = np.empty(moments.shape[:-1])
    rows = []
    for cond_idx, (condition, readout) in enumerate(zip(CONDITIONS, (quiet, noisy))):
        for region_idx, (region, arrivals) in enumerate(zip(regions, truth)):
            for trial_idx, arrival in enumerate(arrivals):
                rng = np.random.default_rng(int(trial_seeds[region_idx, trial_idx]))
                data = read_pulse(readout, arrival, rng)
                s = (weights @ data[..., None]).squeeze(-1)
                moments[cond_idx, region_idx, trial_idx] = s
                estimates[cond_idx, region_idx, trial_idx] = [estimate_arrival(v) for v in s]
            for stage_idx, stage in enumerate(STAGES):
                result = summarize(estimates[cond_idx, region_idx, :, stage_idx], arrivals,
                                   moments[cond_idx, region_idx, :, stage_idx])
                rows.append({"condition": condition, "region": region, "stage": stage, **result})
    np.savez_compressed(
        out_dir / "trials.npz", true_arrival_s=truth, estimated_arrival_s=estimates,
        estimated_moments=moments, trial_seeds=trial_seeds,
        conditions=CONDITIONS, regions=regions, stages=STAGES,
    )
    summary = {
        "detected_pulse_amplitude_pe": 1,
        "seed": seed,
        "trials_per_region": n_trials,
        "regions_ns": REGIONS,
        "known_k": 1,
        "decoder": "rank-1 ESPRIT with 16 consecutive moments",
        "results": rows,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved per-trial results and summary to {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("calibrate", "evaluate"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--trials", type=int, default=500, help="Evaluation trials per arrival region")
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.trials <= 0 or args.seed < 0:
        parser.error("--trials must be positive and --seed must be nonnegative")
    if args.step == "calibrate":
        calibrate(args.out_dir)
    else:
        evaluate(args.out_dir, args.trials, args.seed)


if __name__ == "__main__":
    main()
