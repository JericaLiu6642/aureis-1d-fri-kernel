"""Audit timing vs. remaining record time and run a paired 2×2 window experiment."""
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from single_pulse_fri_baseline import CONDITIONS, DEFAULT_DIR, REGIONS, STAGES, estimate_arrival, read_pulse
from validate_prony_5d_latest import BINS, NS, PERIOD, REPO, RIDGE, RUN_DIR, build_readout, fit_weights, load_experiment, target


SOURCE = DEFAULT_DIR / "trials.npz"
OUT = RUN_DIR / "single_pulse_window_sweep"
WINDOWS = (25, 30)
CAL_ENDS = (20.0, 24.95)
LIMIT_PS = 100
# 1 ns bins in the interior and 0.5 ns bins near the original window edge.
REMAINING_EDGES = np.r_[np.arange(0, 5.5, .5), np.arange(6, 24)]
PLOT_COLORS = ("#16698A", "#A63D22")


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def metrics(errors):
    valid = errors[np.isfinite(errors)]
    return {
        "n": int(errors.size),
        "invalid": int(errors.size - valid.size),
        "p50_abs_ps": float(np.median(valid)) if valid.size else None,
        "p95_abs_ps": float(np.percentile(valid, 95)) if valid.size else None,
        "within_100ps_n": int(np.count_nonzero(errors <= LIMIT_PS)),
        "within_100ps_pct": float(100 * np.mean(errors <= LIMIT_PS)),
    }


def ps_error(estimates, truth):
    return np.abs(estimates - truth) * 1e12


def bin_errors(error, truth, info):
    # Bin labels refer to the original 25 ns endpoint, even for the 30 ns arm,
    # so a bin contains exactly the same events in all four configurations.
    remaining = 25.0 - truth / NS
    bin_index = np.searchsorted(REMAINING_EDGES, remaining, side="right") - 1
    rows = []
    for i, (lo, hi) in enumerate(zip(REMAINING_EDGES[:-1], REMAINING_EDGES[1:])):
        mask = bin_index == i
        if not np.any(mask):
            continue
        rows.append({
            **info, "remaining25_lo_ns": float(lo), "remaining25_hi_ns": float(hi),
            "arrival_lo_ns": float(25 - hi), "arrival_hi_ns": float(25 - lo),
            "median_remaining25_ns": float(np.median(remaining[mask])),
            **metrics(error[mask]),
        })
    return rows


def extended_readouts():
    exp = load_experiment("5d_calorimetry")
    short = build_readout(exp)[2]
    config = deepcopy(exp.HARDWARE)
    config["memory"]["window"] = WINDOWS[-1] * NS
    extended_exp = SimpleNamespace(SIPM=exp.SIPM, HARDWARE=config)
    extended = build_readout(extended_exp)
    # Keep the pole fixed while adding cells. Otherwise increasing depth also
    # increases the modeled bus capacitance, reducing the memory bandwidth.
    memory = extended[2].memory
    unadjusted_bw = memory.bandwidth
    memory.r_on *= memory.bandwidth / short.memory.bandwidth
    assert np.isclose(memory.bandwidth, short.memory.bandwidth, rtol=1e-12)
    assert np.array_equal(extended[2].sample_times()[:short.memory.depth], short.sample_times())

    quiet = deepcopy(extended)
    _, sipm, chain = quiet
    sipm.noise_psd = sipm.ou_drift_std = 0.0
    chain.tia.noise_psd_onesided = chain.tia.temperature = 0.0
    chain.memory.temperature = chain.memory.jitter_rms = 0.0
    chain.adc.noise_rms = 0.0
    meta = {
        "original_hardware": short.params(), "extended_hardware": extended[2].params(),
        "extended_memory_bandwidth_hz_if_original_r_on": unadjusted_bw,
        "extended_internal_span_ns": float(extended[0][-1] / NS),
    }
    return quiet, extended, meta


def build_factorial(truth, seeds, regions):
    quiet, noisy, hardware = extended_readouts()
    sample_times = quiet[2].sample_times()
    n_samples = [int(round(w * NS * quiet[2].memory.samp_rate)) + 1 for w in WINDOWS]
    n_train = [int(round((hi - 2) / .05)) + 1 for hi in CAL_ENDS]
    narrow = np.linspace(2, CAL_ENDS[0], n_train[0])
    extra = np.linspace(CAL_ENDS[0], CAL_ENDS[1], n_train[1] - n_train[0] + 1)[1:]
    train_ns = [narrow, np.r_[narrow, extra]]
    train_y = np.stack([read_pulse(quiet, t * NS, np.random.default_rng(0)) for t in train_ns[-1]])
    y = np.empty((len(CONDITIONS), truth.size, len(STAGES), sample_times.size))
    for ni, readout in enumerate((quiet, noisy)):
        for j, (t, seed) in enumerate(zip(truth, seeds)):
            y[ni, j] = read_pulse(readout, t, np.random.default_rng(int(seed)))
    print(f"Generated {truth.size} matched validation arrivals for each noise condition.", flush=True)

    moments = np.empty((len(WINDOWS), len(CAL_ENDS), *y.shape[:-1], BINS.size), dtype=complex)
    estimates = np.empty(moments.shape[:-1])
    weights = {}
    fit_rows = []
    for wi, (w, count) in enumerate(zip(WINDOWS, n_samples)):
        c_all = np.empty((len(CAL_ENDS), len(STAGES), BINS.size, count), dtype=complex)
        for ci, (hi, times, nt) in enumerate(zip(CAL_ENDS, train_ns, n_train)):
            goal = target(times * NS)
            for si, stage in enumerate(STAGES):
                a = train_y[:nt, si, :count].T
                c = fit_weights(a, goal, RIDGE)
                c_all[ci, si] = c
                fit_rows.append({
                    "window_end_ns": w, "cal_end_ns": hi, "stage": stage,
                    "n_train": nt, "n_waveform_samples": count,
                    "max_weight_row_l2_pe_per_v": float(np.max(np.linalg.norm(c, axis=1))),
                    "training_moment_rms": float(np.sqrt(np.mean(np.abs(c @ a - goal)**2))),
                })
                m = y[:, :, si, :count] @ c.T
                moments[wi, ci, :, :, si] = m
                estimates[wi, ci, :, :, si] = np.array([estimate_arrival(s) for s in m.reshape(-1, BINS.size)]).reshape(m.shape[:2])
        weights[f"C_{w}ns"] = c_all

    np.savez_compressed(OUT / "paired_trials.npz",
        windows_ns=WINDOWS, calibration_ends_ns=CAL_ENDS, conditions=CONDITIONS, stages=STAGES,
        true_arrival_s=truth, trial_seeds=seeds, original_region=regions, estimated_arrival_s=estimates,
        estimated_moments=moments, bins=BINS, period_s=PERIOD,
        waveform_30ns_v=y, sample_times_s=sample_times)
    np.savez_compressed(OUT / "calibration_weights.npz",
        calibration_ends_ns=CAL_ENDS, stages=STAGES, training_arrival_ns=train_ns[-1],
        n_training=n_train, sample_times_s=sample_times, bins=BINS, period_s=PERIOD,
        relative_ridge=RIDGE, **weights)
    write_csv("calibration_diagnostics.csv", fit_rows)
    return estimates, moments, hardware


def analyze_factorial(truth, regions, estimates, moments):
    ideal = target(truth).T
    errors = ps_error(estimates, truth[None, None, None, :, None])
    rows, binned, effects = [], [], []
    # Within-region groups use the same arrival events across configurations.
    for ni, noise in enumerate(CONDITIONS):
        for si, stage in enumerate(STAGES):
            for wi, w in enumerate(WINDOWS):
                for ci, hi in enumerate(CAL_ENDS):
                    info = dict(condition=noise, stage=stage, window_end_ns=w, cal_end_ns=hi)
                    e = errors[wi, ci, ni, :, si]
                    m = moments[wi, ci, ni, :, si]
                    binned.extend(bin_errors(e, truth, info))
                    for region in REGIONS:
                        mask = regions == region
                        rows.append({
                            **info, "original_region": region, **metrics(e[mask]),
                            "median_relative_moment_error": float(np.median(np.linalg.norm(m[mask] - ideal[mask], axis=1) / np.sqrt(BINS.size))),
                        })
            for region in REGIONS:
                mask = regions == region
                # Signed changes are relative to the same events in the reference arm.
                comparisons = [
                    ("widen_calibration", f"window={w} ns", (wi, 0), (wi, 1))
                    for wi, w in enumerate(WINDOWS)
                ] + [
                    ("extend_readout", f"calibration_end={hi:g} ns", (0, ci), (1, ci))
                    for ci, hi in enumerate(CAL_ENDS)
                ]
                for factor, held, (wa, ca), (wb, cb) in comparisons:
                    a = errors[wa, ca, ni, mask, si]
                    b = errors[wb, cb, ni, mask, si]
                    ma, mb = metrics(a), metrics(b)
                    effects.append({
                        "condition": noise, "stage": stage, "original_region": region,
                        "factor": factor, "held_fixed": held, "n": len(a),
                        "before_p95_ps": ma["p95_abs_ps"], "after_p95_ps": mb["p95_abs_ps"],
                        "before_within_100ps_pct": ma["within_100ps_pct"], "after_within_100ps_pct": mb["within_100ps_pct"],
                        "crossed_from_fail_to_pass": int(np.count_nonzero((~(a <= LIMIT_PS)) & (b <= LIMIT_PS))),
                        "crossed_from_pass_to_fail": int(np.count_nonzero((a <= LIMIT_PS) & (~(b <= LIMIT_PS)))),
                    })
    write_csv("paired_summary.csv", rows)
    write_csv("paired_binned.csv", binned)
    write_csv("paired_contrasts.csv", effects)
    return binned


def boundary_rows(bins):
    result = []
    for condition in CONDITIONS:
        for stage in STAGES:
            group = [r for r in bins if r["condition"] == condition and r["stage"] == stage]
            group.sort(key=lambda r: r["arrival_lo_ns"])
            fails = [r["p95_abs_ps"] > LIMIT_PS for r in group]
            index = next((i for i, failed in enumerate(fails) if failed), None)
            sustained = next((i for i in range(len(fails)) if all(fails[i:])), None)
            result.append({
                "condition": condition, "stage": stage,
                "first_failing_bin": group[index] if index is not None else None,
                "first_bin_of_contiguous_failing_suffix": group[sustained] if sustained is not None else None,
            })
    return result


def original_chart(truth, err, rows):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True, layout="constrained")
    x = 25 - truth / NS
    for ni, noise in enumerate(CONDITIONS):
        for si, stage in enumerate(STAGES):
            ax = axes[ni, si]
            ax.scatter(x, np.maximum(err[ni, :, si], 1e-5), color=PLOT_COLORS[si], alpha=.21, s=7, lw=0, label="Individual trials")
            group = [r for r in rows if r["condition"] == noise and r["stage"] == stage]
            ax.plot([r["median_remaining25_ns"] for r in group], [r["p95_abs_ps"] for r in group],
                    color=PLOT_COLORS[si], marker="o", ms=3.5, lw=1.6, label="P95 per bin")
            ax.axhline(LIMIT_PS, color="#343B44", ls="--", lw=1, label="100 ps threshold")
            ax.axvline(5, color="#8592A0", ls=":", lw=1.2, label="Calibration end")
            ax.set(title=f"{('TIA', 'ADC')[si]} · {noise.replace('_', ' ')}", yscale="log",
                   xlim=(23.2, -.1), ylim=(1e-5, 4e4), xticks=(23, 20, 15, 10, 5, 2.5, 0))
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.supylabel("Absolute timing error (ps)")
    fig.supxlabel("Time remaining before the original 25 ns endpoint (ns)")
    fig.suptitle("Existing trials · Where does timing error exceed 100 ps?")
    fig.savefig(OUT / "original_timing_vs_remaining.png", dpi=200)
    plt.close(fig)


def factorial_chart(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.115, top=.825, wspace=.10, hspace=.23)
    for ni, noise in enumerate(CONDITIONS):
        for si, stage in enumerate(STAGES):
            ax = axes[ni, si]
            for wi, w in enumerate(WINDOWS):
                for ci, hi in enumerate(CAL_ENDS):
                    group = [r for r in rows if (r["condition"], r["stage"], r["window_end_ns"], r["cal_end_ns"]) == (noise, stage, w, hi)
                             and r["remaining25_hi_ns"] <= 8]
                    ax.plot([r["median_remaining25_ns"] for r in group], [r["p95_abs_ps"] for r in group],
                            color=PLOT_COLORS[wi], ls=("-", "--")[ci], marker=("o", "s")[ci],
                            ms=4, lw=1.6, label=f"Read to {w} ns · train to {hi:g} ns")
            ax.axhline(LIMIT_PS, color="#343B44", ls=":", lw=1.2)
            ax.axvline(5, color="#8592A0", ls=":", lw=1.2)
            ax.set(title=f"{('TIA', 'ADC')[si]} · {noise.replace('_', ' ')}", yscale="log",
                   xlim=(8, -.05), ylim=(.01, 4e4), xticks=(8, 7, 6, 5, 4, 3, 2, 1, 0))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=2, frameon=False, fontsize=9)
    fig.supylabel("P95 absolute timing error per bin (ps)")
    fig.supxlabel("Time remaining before the original 25 ns endpoint (ns) · 30 ns arm adds 5 ns")
    fig.suptitle("Paired experiment · Calibration range and capture window", y=.984)
    fig.savefig(OUT / "paired_calibration_vs_window.png", dpi=200)
    plt.close(fig)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Baseline trials missing at {SOURCE}. Run baseline calibrate and evaluate first.")
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE) as f:
        assert tuple(f["conditions"]) == CONDITIONS and tuple(f["stages"]) == STAGES
        assert tuple(f["regions"]) == tuple(REGIONS)
        truth = f["true_arrival_s"].reshape(-1)
        regions = np.repeat(f["regions"], f["true_arrival_s"].shape[1])
        seeds = f["trial_seeds"].reshape(-1)
        original = f["estimated_arrival_s"].reshape(len(CONDITIONS), truth.size, len(STAGES))
    err = ps_error(original, truth[None, :, None])
    original_bins = [
        row for ni, noise in enumerate(CONDITIONS) for si, stage in enumerate(STAGES)
        for row in bin_errors(err[ni, :, si], truth, dict(condition=noise, stage=stage))
    ]
    write_csv("original_binned.csv", original_bins)
    original_chart(truth, err, original_bins)
    estimates, moments, hardware = build_factorial(truth, seeds, regions)
    new_bins = analyze_factorial(truth, regions, estimates, moments)
    factorial_chart(new_bins)
    meta = {
        "source": SOURCE.relative_to(RUN_DIR).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "simulator_dir": REPO.name,
        "simulator_git_head": subprocess.run(
            ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "n_validation": len(truth), "input_pulse_pe": 1, "known_k": 1,
        "criterion": "Bin P95 absolute linear arrival error > 100 ps is above target. Empirical percentiles use finite estimates.",
        "remaining25_bin_edges_ns": REMAINING_EDGES.tolist(), "calibration_ends_ns": CAL_ENDS,
        "readout_window_ends_ns": WINDOWS, "training_spacing_ps": 50, "period_ns": PERIOD / NS,
        "moment_indices": BINS.tolist(), "relative_ridge": RIDGE, "adc_quantization": "enabled in all arms",
        "pairing": "Reuse original validation arrival times and seeds. Within each event/noise setting, both capture windows are prefixes of one 30 ns readout. This also pairs all overlapping noise samples. The original archived 25 ns run is separate.",
        "hardware": hardware, "original_bin_boundaries": boundary_rows(original_bins),
        "paired_npz_axes": {
            "estimated_arrival_s": ["window", "calibration_end", "condition", "event", "stage"],
            "estimated_moments": ["window", "calibration_end", "condition", "event", "stage", "moment"],
            "waveform_30ns_v": ["condition", "event", "stage", "sample"],
            "calibration_C": ["calibration_end", "stage", "moment", "sample"],
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved diagnostics to {OUT}", flush=True)


if __name__ == "__main__":
    main()
