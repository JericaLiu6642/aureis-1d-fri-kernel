# Aureis 1D FRI Kernel

Simulation and analysis code for the AUREIS 5D calorimeter SiPM readout. The experiments ask whether fixed sampling weights can reproduce exponential moments from a simulated pulse waveform, and how noise, quantization, record length and pulse overlap affect arrival-time recovery.

The project uses the NumPy backend of the [SLAC AUREIS 1D digital twin](https://github.com/slaclab/aureis-1d-twin) at commit [`68a42b9e76a566cde9c7cd9775859936a0adce55`](https://github.com/slaclab/aureis-1d-twin/commit/68a42b9e76a566cde9c7cd9775859936a0adce55). The upstream repository is private. A GitHub account with upstream access is required to obtain the simulator. This repository contains the analysis scripts and synthetic results. The simulator is a separate checkout.

## Setup

Use Python 3.11. The reference runs used Python 3.11.9, NumPy 2.4.6 and Matplotlib 3.11.2. The core experiments use NumPy. Matplotlib generates the figures. Git records the simulator revision in the window-sweep metadata.

From the directory containing this repository, clone the simulator beside it if you do not already have the pinned checkout. A sparse checkout is sufficient and avoids downloading the C/S event datasets.

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --filter=blob:none --sparse https://github.com/slaclab/aureis-1d-twin.git aureis-1d-twin-latest
git -C aureis-1d-twin-latest sparse-checkout set src experiments/5d_calorimetry
git -C aureis-1d-twin-latest checkout --detach 68a42b9e76a566cde9c7cd9775859936a0adce55
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location aureis-1d-fri-kernel

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS or Linux, use `python3.11 -m venv .venv` and `./.venv/bin/python -m pip install -r requirements.txt`. The Python commands below use the name `python`. Activate the virtual environment or substitute its interpreter path. The analysis scripts load the twin from its checkout, so an editable package install is optional.

| Environment variable | Default | Use |
|---|---|---|
| `AUREIS_TWIN_DIR` | Sibling `aureis-1d-twin-latest` | Point at an existing checkout of the pinned simulator revision. |
| `FRI_RUN_DIR` | `runs/` in this repo | Choose an output root for a separate run. Use the same root throughout the workflow. |
| `AUREIS_TWIN_LEGACY_DIR` | Sibling `aureis-1d-twin-source` | Select the original simulator snapshot for the archival script only. |

## Run the simulations

Run these commands from the repository root.

```text
python scripts/single_pulse_fri_baseline.py calibrate
python scripts/single_pulse_fri_baseline.py evaluate --trials 500 --seed 20261108
python scripts/single_pulse_noise_ablation.py
python scripts/diagnose_noise_sensitivity.py
python scripts/single_pulse_window_sweep.py
python scripts/plot_single_pulse.py
python scripts/validate_prony_5d_latest.py
```

| Script | Inputs and purpose | Files written under `runs/` |
|---|---|---|
| [`single_pulse_fri_baseline.py`](scripts/single_pulse_fri_baseline.py), `calibrate` | Inject 361 single-photoelectron pulses at 50 ps spacing from 2 to 20 ns. Fit separate fixed weights for the TIA and ADC taps. | `single_pulse_5d/weights.npz` |
| Same script, `evaluate` | Load the saved weights. Recover arrivals for 500 random events in each region using rank-1 ESPRIT. | `single_pulse_5d/trials.npz`, `single_pulse_5d/summary.json` |
| [`single_pulse_noise_ablation.py`](scripts/single_pulse_noise_ablation.py) | Reuse the in-range trials and weights. Switch on one noise source at a time and verify the quiet and full-noise controls against the baseline. | `single_pulse_5d/noise_ablation_2_20ns.csv` |
| [`diagnose_noise_sensitivity.py`](scripts/diagnose_noise_sensitivity.py) | Compare ridge settings from 0.001 to 0.3 and compare ESPRIT with fixed-amplitude template matching. Recheck the 0.001 versus 0.01 comparison on an independent set of 500 arrivals. | `single_pulse_5d/noise_sensitivity.json` |
| [`single_pulse_window_sweep.py`](scripts/single_pulse_window_sweep.py) | Reuse baseline arrivals and seeds. Compare 25 and 30 ns acquisitions and calibration endpoints of 20 and 24.95 ns. | `single_pulse_window_sweep/` tables, metadata, weights, trial arrays and figures |
| [`plot_single_pulse.py`](scripts/plot_single_pulse.py) | Plot the sensor current, TIA pulse, 25 ns cutoff and arrival-error distributions. Print rise time, FWHM and the last 1% crossing. | `single_pulse_5d/figures/` |
| [`validate_prony_5d_latest.py`](scripts/validate_prony_5d_latest.py) | Run the direct moment-reproduction check, a finite-window Fourier comparison and a controlled two-pulse pilot. It also scores moment error on the bundled 100 synthetic 5D events. | `results_5d_latest.json` |

The direct moment and two-pulse pilot is independent of the single-pulse baseline. The noise, window and plotting scripts read from the default baseline subdirectory under `FRI_RUN_DIR`. To keep two runs separate, set `FRI_RUN_DIR` before calibrating and use that value for all the dependent commands. The baseline also accepts `--out-dir` on either subcommand when run on its own.

## Signal model and calibration

The simulator generates the current from detected pulses, filters it through the TIA, samples it in analog memory and quantizes the samples with the ADC. For the controlled studies a pulse is injected directly as a detected avalanche, with known time and amplitude. The default 5D photon generator is used only for the bundled synthetic event sample.

| Readout block | Baseline setting |
|---|---|
| SiPM | onsemi MicroFJ-30020, 3 × 3 mm² fast output |
| TIA | 30 mW initialization, 1 kΩ transimpedance, two low-pass poles at 1 GHz each |
| Analog memory | 0–25 ns, 251 nominal samples at 10 GS/s, 4.5 GHz model bandwidth |
| ADC | 8 bits, 0.7 V span, 2.734375 mV/code |
| Internal simulation time step | 5 ps |

The TIA tap is interpolated to the 251 nominal sample positions. The ADC tap uses code-center voltages with the modeled baseline subtracted. The taps get separate calibration weights. In the earlier two-pulse pilot the ADC arm reuses the memory weights to expose the effect of quantization. That differs from the later, independently calibrated ADC baseline.

For a 1 p.e. calibration pulse at time $\tau_j$, the target matrix has entries

$$
E_{m,j}=\exp(-i2\pi m\tau_j/T),\qquad
m=1,\ldots,16,\qquad T=25\ \mathrm{ns}.
$$

Each column of $A$ contains one waveform, and the matching column of $E$ contains its target moments. With 361 calibration events, $A$ is $251\times361$, $E$ is $16\times361$, and $C$ is $16\times251$. The weights minimize $\|CA-E\|_F^2+\lambda^2\|C\|_F^2$ with $\lambda=10^{-3}\sigma_{\max}(A)$ by default. The validation stage loads $C$ and computes $\mathbf{s}=C\mathbf{y}$ for each new waveform. ESPRIT estimates the arrival from the phase progression across the moments. Validation truth is used to score the recovery.

## How the experiments differ

- **Single pulse.** Use the same 500 uniform arrivals in 2–20 ns and 500 uniform arrivals in 20.025–24.925 ns for both taps and noise conditions. Calibration covers 2–20 ns. For a late pulse the record can end before the full peak is captured. ADC quantization is enabled in every condition.
- **Noise ablation.** Start from the quiet chain and enable one of `sipm.noise_psd`, `tia.noise_psd_onesided`, `tia.temperature`, `memory.jitter_rms`, `memory.temperature` or `adc.noise_rms`. The amplifier-only test uses a rounded PSD of $1.47\times10^{-20}$ A²/Hz. The all-noise control uses the simulator default at full precision. The SiPM white-current noise value is a placeholder in the upstream model.
- **Window sweep.** The 25 ns record is the prefix of the same 30 ns waveform for each condition, so the two arms share the overlapping samples and noise. The memory bandwidth is held fixed in this controlled test. Keep the capture window and the calibration range separate when interpreting the results.
- **Two-pulse pilot.** Compare separations of 0.3, 1.3 and 3.0 ns and equal amplitudes of 1 or 5 p.e. per pulse. The first arrival is fixed at 6.037 ns. The decoder is given $K=2$. This pilot probes overlap and the effect of noise. A minimum resolvable spacing across random arrivals and amplitude ratios remains a follow-up.
- **Archival pilot.** [`legacy/validate_prony.py`](legacy/validate_prony.py) preserves the first analysis against the earlier local twin. Run `python legacy/validate_prony.py` with the original source snapshot selected via `AUREIS_TWIN_LEGACY_DIR`. The exact upstream Git revision of that older export is unrecorded. The archived [`results.json`](reference_results/legacy/results.json) stores SHA-256 hashes of its simulator Python files for provenance.

## Reference results

[`reference_results/`](reference_results/) contains the compact tables and metadata from the simulation studies. The larger per-event arrays, weights and plots are regenerated under `runs/`. [`data/synthetic_5d/`](data/synthetic_5d/) contains the simulated inputs and event records for the pilot checks. The current 100-event sample comes from the twin's placeholder photon generator with seed 1.

| Single-pulse condition | TIA P95 absolute error | ADC P95 absolute error | Within 100 ps, TIA / ADC |
|---|---:|---:|---:|
| 2–20 ns, no added noise | 0.033 ps | 17.6 ps | 100% / 100% |
| 2–20 ns, default noise | 544 ps | 13,819 ps | 32% / 17.4% |
| 20.025–24.925 ns, no added noise | 3,354 ps | 4,695 ps | 7% / 4% |
| 20.025–24.925 ns, default noise | 19,642 ps | 8,474 ps | 3% / 0.4% |

P95 is computed over finite estimates using the absolute, linear time error $|\hat{\tau}-\tau|$. The within-100 ps fraction uses all trials. An invalid estimate counts as a failure for that fraction. ESPRIT returns times modulo 25 ns, and scoring uses the time difference in the physical record.

The in-range ablation identified TIA amplifier noise as the largest perturbation in the current model. Raising the relative ridge from 0.001 to 0.01 reduced ADC P95 from 13.82 ns to about 0.63 ns on the same noisy waveforms. Full-waveform matching with a fixed 1 p.e. template is an exploratory benchmark for how much timing information remains in those samples. These numbers describe this simulator, preset and evaluation protocol.

### Main NumPy artifacts

| Artifact | Contents |
|---|---|
| `weights.npz` | `tia_nominal` and `adc_v` complex $16\times251$ matrices, nominal sample times, calibration times, frequency-bin indices, moment period, relative ridge and hardware metadata |
| `trials.npz` | `true_arrival_s` and `trial_seeds` with axes `[region, trial]`, `estimated_arrival_s` with axes `[condition, region, trial, stage]`, and `estimated_moments` with an added final moment axis |
| Window sweep `paired_trials.npz` | Per-event waveforms, estimates and moments for the paired window comparison. `metadata.json` describes the axes and hardware adjustments. |
| Synthetic 5D `5d_latest_100events.npz` | 100×251 ADC codes, nominal sample times, event IDs and detected-pulse times from the placeholder generator. |

## Follow-up experiments

Extend the two-pulse study to random offsets, amplitude ratios and unknown pulse counts. Compare digital Fourier extraction with calibrated moments using the same decoder. Add an explicit post-TIA I/Q extractor and compare timing versus conversion count. Sweep noise and dynamic range before comparing Prony/ESPRIT and OMP on the same events and frequency sets.

## 中文重跑提示

先取得指定版本的 SLAC 模擬器，再執行 `calibrate` 和 `evaluate`。逐項雜訊、讀出窗與繪圖程式會讀取這一輪的權重及試驗資料。`reference_results/` 是已整理的結果，`runs/` 是本機重跑輸出。若要改模型設定，請為該組設定重新校準，再用同一批到達時間比較。
