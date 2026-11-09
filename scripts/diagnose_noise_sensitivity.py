"""Re-run the exploratory ridge and decoder checks on paired in-range trials."""

import json

import numpy as np

from single_pulse_fri_baseline import DEFAULT_DIR, STAGES, read_pulse, setup, summarize
from single_pulse_noise_ablation import collect_waveforms, in_range_trials, recover
from validate_prony_5d_latest import NS, fit_weights, target


RIDGE_VALUES = (0.001, 0.01, 0.03, 0.1, 0.3)
PULSE_TEMPLATE_GRID = np.arange(500) * 50e-12


def main():
    arrivals, seeds, _, _ = in_range_trials()
    with np.load(DEFAULT_DIR / "weights.npz") as artifact:
        baseline_weights = np.stack([artifact[stage] for stage in STAGES])
        train = artifact["train_times_s"]

    quiet, _ = setup(False)
    noisy, _ = setup(True)
    reference = target(train)
    clean_waveforms = collect_waveforms(quiet, arrivals, seeds)
    noisy_waveforms = collect_waveforms(noisy, arrivals, seeds)
    conditions = (("no_added_noise", clean_waveforms), ("default_noise", noisy_waveforms))
    train_waveforms = collect_waveforms(quiet, train, np.zeros(train.size, dtype=int))
    pulse_templates = collect_waveforms(
        quiet, PULSE_TEMPLATE_GRID, np.zeros(PULSE_TEMPLATE_GRID.size, dtype=int)
    )
    exponential_templates = target(PULSE_TEMPLATE_GRID)
    rows = []

    for condition, waveforms in conditions:
        moments = (baseline_weights @ waveforms[..., None]).squeeze(-1)
        for j, stage in enumerate(STAGES):
            # Fixed amplitude of 1 p.e. in both dictionaries.
            templates = pulse_templates[:, j]
            waveform_score = waveforms[:, j] @ templates.T - np.sum(templates**2, axis=1) / 2
            moment_score = (moments[:, j] @ exponential_templates.conj()).real
            for method, score in (("waveform_template", waveform_score), ("moment_template", moment_score)):
                estimate = PULSE_TEMPLATE_GRID[np.argmax(score, axis=1)]
                result = summarize(estimate, arrivals, moments[:, j])
                # These rows compare arrival estimates only.
                result.pop("median_moment_relative_error")
                rows.append({"condition": condition, "stage": stage, "method": method, **result})

    ridge_weights = {}
    for ridge in RIDGE_VALUES:
        weights = np.stack([fit_weights(train_waveforms[:, j].T, reference, ridge)
                            for j in range(len(STAGES))])
        ridge_weights[ridge] = weights
        for condition, waveforms in conditions:
            moments = (weights @ waveforms[..., None]).squeeze(-1)
            estimates = recover(moments)
            for j, stage in enumerate(STAGES):
                rows.append({
                    "condition": condition, "stage": stage, "method": "rank_1_esprit",
                    "relative_ridge": ridge, "weight_frobenius_norm": float(np.linalg.norm(weights[j])),
                    **summarize(estimates[:, j], arrivals, moments[:, j]),
                })

    # A separate holdout run verifies that the 0.01 ridge comparison is not
    # specific to the original 500 arrivals or their noise draws.
    rng = np.random.default_rng(11204)
    extra_arrivals = rng.uniform(2, 20, 500) * NS
    extra = np.stack([read_pulse(noisy, tau, rng) for tau in extra_arrivals])
    for ridge in (0.001, 0.01):
        weights = baseline_weights if ridge == 0.001 else ridge_weights[ridge]
        moments = (weights @ extra[..., None]).squeeze(-1)
        estimates = recover(moments)
        rows.append({
            "condition": "default_noise_independent_seed_11204", "stage": "adc_v",
            "method": "rank_1_esprit", "relative_ridge": ridge,
            **summarize(estimates[:, 1], extra_arrivals, moments[:, 1]),
        })

    path = DEFAULT_DIR / "noise_sensitivity.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
