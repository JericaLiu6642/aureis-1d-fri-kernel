"""Repeat the one-at-a-time 5D noise test using saved single-pulse trials."""

import csv

import numpy as np

from single_pulse_fri_baseline import DEFAULT_DIR, STAGES, estimate_arrival, read_pulse, setup, summarize


CASES = (
    ("No added noise", "", None, ""),
    ("SiPM white current", "sipm.noise_psd", 3.2e-23, "A^2/Hz"),
    ("TIA amplifier", "tia.noise_psd_onesided", 1.47e-20, "A^2/Hz"),
    ("TIA resistor thermal", "tia.temperature", 300.0, "K"),
    ("Memory sampling jitter", "memory.jitter_rms", 1e-12, "s RMS"),
    ("Memory kBT/C", "memory.temperature", 300.0, "K"),
    ("ADC pre-quantization", "adc.noise_rms", 1e-4, "V RMS"),
    ("All default noise", "5D readout defaults", None, ""),
)


def in_range_trials():
    with np.load(DEFAULT_DIR / "trials.npz") as source:
        inside = source["regions"].tolist().index("inside")
        return (
            source["true_arrival_s"][inside], source["trial_seeds"][inside],
            source["estimated_arrival_s"][:, inside], source["estimated_moments"][:, inside],
        )


def collect_waveforms(readout, arrivals, seeds):
    return np.stack([read_pulse(readout, tau, np.random.default_rng(int(seed)))
                     for tau, seed in zip(arrivals, seeds)])


def recover(moments):
    return np.array([[estimate_arrival(m) for m in stage]
                     for stage in moments.transpose(1, 0, 2)]).T


def main():
    truth, seeds, reference_estimates, reference_moments = in_range_trials()
    with np.load(DEFAULT_DIR / "weights.npz") as artifact:
        weights = np.stack([artifact[stage] for stage in STAGES])
        ridge = artifact["relative_ridge"].item()

    rows = []
    for case, parameter, value, unit in CASES:
        default_noise = case == "All default noise"
        readout, _ = setup(noise=default_noise)
        if value is not None:
            block, attr = parameter.split(".")
            component = readout[1] if block == "sipm" else getattr(readout[2], block)
            setattr(component, attr, value)

        waveforms = collect_waveforms(readout, truth, seeds)
        moments = (weights @ waveforms[..., None]).squeeze(-1)
        estimates = recover(moments)
        if value is None:
            control = int(default_noise)
            np.testing.assert_allclose(moments, reference_moments[control], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(estimates, reference_estimates[control], atol=1e-17, rtol=1e-8)

        for j, stage in enumerate(STAGES):
            result = summarize(estimates[:, j], truth, moments[:, j])
            rows.append({
                "noise_condition": case, "enabled_parameter": parameter,
                "parameter_value": value, "parameter_unit": unit,
                "arrival_min_ns": 2, "arrival_max_ns": 20,
                "pulse_amplitude_pe": 1, "relative_ridge": ridge, "readout_stage": stage,
                "trials": result["trials"], "invalid_estimates": result["invalid_estimates"],
                "median_absolute_error_ps": result["median_absolute_error_ps"],
                "p95_absolute_error_ps": result["p95_absolute_error_ps"],
                "within_100ps_pct": 100 * result["within_100ps"],
                "median_moment_relative_error_pct": 100 * result["median_moment_relative_error"],
            })

    path = DEFAULT_DIR / "noise_ablation_2_20ns.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}. Both controls reproduce the baseline trials.")


if __name__ == "__main__":
    main()
