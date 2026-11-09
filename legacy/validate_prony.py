"""Small reproducible validation of moments from the unmodified local twin.

Run this archival pilot with the original simulator snapshot (see README).
Only numpy and matplotlib are required. Ground truth is used for scoring,
never to form measured moments. K=2 is specified by this controlled experiment.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

chinese_font = Path('C:/Windows/Fonts/msjh.ttc')
if chinese_font.exists():
    font_manager.fontManager.addfont(str(chinese_font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(chinese_font)).get_name()
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('AUREIS_TWIN_LEGACY_DIR', ROOT.parent / 'aureis-1d-twin-source')).resolve()
if not (REPO / 'experiments' / '5d_calorimetry' / 'experiment.py').is_file():
    raise SystemExit(f"Original simulator snapshot missing at {REPO}. Set AUREIS_TWIN_LEGACY_DIR (see README).")
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'experiments'))
from run import load_experiment, build_readout

OUT = Path(os.environ.get('FRI_RUN_DIR', ROOT / 'runs')).resolve() / 'legacy'
OUT.mkdir(parents=True, exist_ok=True)
t, sipm, chain = build_readout(load_experiment('5d_calorimetry'))
original_noise = (sipm.noise_psd, chain.tia.noise_psd_onesided,
                  chain.memory.temperature, chain.memory.jitter_rms, chain.adc.noise_rms)
T = 25e-9
m = np.arange(1, 17)  # consecutive bins; avoid the fast-output DC zero
freq = m / T


def noise(enabled):
    values = original_noise if enabled else (0., 0., 0., 0., 0.)
    (sipm.noise_psd, chain.tia.noise_psd_onesided, chain.memory.temperature,
     chain.memory.jitter_rms, chain.adc.noise_rms) = values


def forward(times, amps=None, seed=0):
    rng = np.random.default_rng(seed)
    sipm.rng = rng
    current = sipm.simulate(np.asarray(times), amps=amps)[1]
    out = chain.simulate(current, rng)
    volts = chain.adc.to_volts(out['adc_out']) - chain.adc.baseline
    return out['memory_v'], volts, out['adc_out']


def truth(times, amps):
    return np.exp(-2j * np.pi * freq[:, None] * np.asarray(times)) @ np.asarray(amps)


def hankel(y):
    return np.array([y[i:i+9] for i in range(8)])


def estimate(y, K=2):
    H = hankel(y)
    U, s, _ = np.linalg.svd(H[:, :-1], full_matrices=False)
    # Shift invariance of the left Vandermonde subspace (ESPRIT).
    Us = U[:, :K]
    roots = np.linalg.eigvals(np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0])
    times = np.sort(np.mod(-np.angle(roots) * T / (2*np.pi), T))
    V = np.exp(-2j*np.pi * freq[:, None] * times)
    amps = np.linalg.lstsq(V, y, rcond=None)[0]
    return times, amps, roots


def score(y, times, amps):
    target = truth(times, amps)
    tt, aa, roots = estimate(y)
    sv = np.linalg.svd(hankel(y), compute_uv=False)
    return dict(moment_rel=float(np.linalg.norm(y-target)/np.linalg.norm(target)),
                time_rmse_ps=float(np.sqrt(np.mean((tt-times)**2))*1e12),
                max_time_error_ps=float(np.max(np.abs(tt-times)))*1e12,
                amplitude_rel=float(np.linalg.norm(aa-amps)/np.linalg.norm(amps)),
                sigma3_over_sigma2=float(sv[2]/sv[1]),
                max_root_radius_error=float(np.max(np.abs(np.abs(roots)-1))))


noise(False)
# Independent, known single-photon calibration, not evaluation-event truth.
cal_times = np.arange(2.0, 20.0001, .05) * 1e-9
A = np.column_stack([forward([tau])[0] for tau in cal_times])
E = np.exp(-2j*np.pi*freq[:, None]*cal_times)
U, sv, Vh = np.linalg.svd(A, full_matrices=False)
hold_times = np.arange(2.025, 19.999, .1) * 1e-9
A_hold = np.column_stack([forward([tau])[0] for tau in hold_times])
E_hold = np.exp(-2j*np.pi*freq[:, None]*hold_times)

weights = {}
calibration = []
for r in (1e-2, 1e-3, 1e-4, 1e-5):
    # C A approximates E; ridge level is r * largest singular value of A.
    C = ((E @ Vh.conj().T) * (sv/(sv**2+(r*sv[0])**2)) ) @ U.conj().T
    weights[r] = C
    err = C @ A_hold - E_hold
    calibration.append(dict(ridge_relative=r,
                            heldout_rms=float(np.sqrt(np.mean(np.abs(err)**2))),
                            max_row_weight_norm=float(np.linalg.norm(C, axis=1).max())))

# Preselected regularisation, fixed before event tests (not fitted to their truth).
ridge = 1e-3
C = weights[ridge]
# Naive finite-window Fourier division: a known isolated template at time zero.
ts = chain.sample_times()
dt_s = 1/chain.memory.samp_rate
quadrature = np.ones(len(ts))*dt_s
quadrature[[0,-1]] *= .5
F = np.exp(-2j*np.pi*freq[:,None]*ts)*quadrature
template = forward([0.])[0]
G = F @ template

rows = []
for separation_ns in (.3, 1.3, 3.0):
    times = np.array([6.037, 6.037+separation_ns])*1e-9
    for amp in (1., 5.):
        amps = np.array([amp, amp])
        noise(False)
        raw, adc, codes = forward(times, amps)
        for name, y in [('oracle_control', truth(times, amps)),
                        ('naive_fourier_memory_no_noise', (F@raw)/G),
                        ('calibrated_memory_no_noise', C@raw),
                        ('calibrated_adc_quantization_only', C@adc)]:
            rows.append(dict(separation_ns=separation_ns, amplitude_each_pe=amp,
                             method=name, seed=None, **score(y, times, amps),
                             rail_fraction=float(np.mean((codes==0)|(codes==chain.adc.levels-1)))))
        noise(True)
        for seed in range(50):
            raw, adc, codes = forward(times, amps, seed)
            rows.append(dict(separation_ns=separation_ns, amplitude_each_pe=amp,
                             method='calibrated_adc_all_default_noise', seed=seed,
                             **score(C@adc, times, amps),
                             rail_fraction=float(np.mean((codes==0)|(codes==chain.adc.levels-1)))))

summary = []
for sep in (.3, 1.3, 3.0):
    for amp in (1., 5.):
        for method in dict.fromkeys(r['method'] for r in rows):
            subset = [r for r in rows if r['separation_ns']==sep and
                      r['amplitude_each_pe']==amp and r['method']==method]
            summary.append(dict(separation_ns=sep, amplitude_each_pe=amp, method=method,
                n=len(subset), **{key:float(np.median([r[key] for r in subset]))
                for key in ('moment_rel','time_rmse_ps','amplitude_rel','sigma3_over_sigma2','rail_fraction')},
                success_both_within_100ps=float(np.mean([r['max_time_error_ps']<100
                                                       for r in subset]))))

data = np.load(ROOT / 'data' / 'synthetic_5d' / 'legacy_5d.npz')
counts = np.array([np.count_nonzero(data['detected_event']==i) for i in data['event_id']])
metadata = dict(package_version='0.2.0', numpy_version=np.__version__,
    t_step_ps=float(np.median(np.diff(t))*1e12), T_ns=T*1e9, bins=m.tolist(),
    tia_poles_GHz=(np.asarray(chain.tia.poles)/1e9).tolist(),
    original_noise=list(original_noise), memory_bandwidth_GHz=chain.memory.bandwidth/1e9,
    adc_lsb_mV=chain.adc.lsb*1e3, calibration_interval_ns=[2.,20.],
    hardware=dict(tia=asdict(chain.tia), memory=asdict(chain.memory), adc=asdict(chain.adc)),
    calibration_step_ps=50., heldout_offset_ps=25., chosen_relative_ridge=ridge,
    existing_output=dict(events=len(counts), sample_shape=list(data['adc_out'].shape),
                         mean_detected=float(counts.mean()), min_detected=int(counts.min()),
                         max_detected=int(counts.max())),
    source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (REPO/'src').rglob('*.py')})
payload = dict(metadata=metadata, calibration=calibration, summary=summary, trials=rows,
               caution='Only 50 seeds per condition; fixed K=2; no general recovery guarantee.')
(OUT/'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
np.savez_compressed(OUT/'calibration.npz', weights=C, bins=m, period=T,
                    sample_times=ts, calibration_times=cal_times)
for r in summary:
    if r['separation_ns']==1.3:
        print(json.dumps(r))
print('Calibration:', json.dumps(calibration))

times = np.array([6.037,7.337])*1e-9
noise(False)
raw, _, _ = forward(times, [1.,1.])
target = truth(times,[1.,1.])
fig, axes = plt.subplots(1,3,figsize=(15,4.0))
axes[0].plot(ts*1e9, raw*1e3)
axes[0].vlines(times*1e9, 0, raw.max()*1e3, colors='tab:orange', linestyles='--')
axes[0].set(xlabel='時間 (ns)',ylabel='電壓 (mV)',title='無雜訊：兩個單光子脈衝')
axes[1].plot(m,np.abs((F@raw)/G-target),'o-',label='有限窗 Fourier 除頻譜')
axes[1].plot(m,np.abs(C@raw-target),'o-',label='校準後的取樣權重')
axes[1].set(xlabel='Moment 索引 m',ylabel='Moment 絕對誤差',yscale='log',title='離網格到達時間；間距 1.3 ns')
axes[1].legend(fontsize=9)
for key, label in [('oracle_control','真值 moments'),('naive_fourier_memory_no_noise','Fourier 除頻譜'),
                   ('calibrated_memory_no_noise','校準權重')]:
    y = {'oracle_control':target, 'naive_fourier_memory_no_noise':(F@raw)/G,
         'calibrated_memory_no_noise':C@raw}[key]
    s=np.linalg.svd(hankel(y),compute_uv=False)
    axes[2].semilogy(np.arange(1,len(s)+1),s/s[0],'o-',label=label)
axes[2].set(xlabel='奇異值索引',ylabel='正規化奇異值',title='Hankel 矩陣：理想秩為 K=2')
axes[2].legend(fontsize=9)
for ax in axes[1:]:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:.0e}'))
fig.tight_layout()
fig.savefig(OUT/'validation.png',dpi=180)
print('Saved results.json, calibration.npz, validation.png')
