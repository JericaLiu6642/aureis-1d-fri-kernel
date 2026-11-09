"""Regenerate waveform, cutoff and timing plots from baseline trials."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from single_pulse_fri_baseline import DEFAULT_DIR, setup
from validate_prony_5d_latest import NS
from aureis_twin.analog.tia import filter_analog

data_dir = DEFAULT_DIR
figure_dir = data_dir / 'figures'
figure_dir.mkdir(parents=True, exist_ok=True)
(t, sipm, chain), hardware = setup(False)

def waveform(arrival):
    _, current = sipm.simulate([arrival*NS], amps=[1.0])
    return chain.simulate(current, np.random.default_rng(0))['tia_v']

def shape_metrics(x, y):
    y = y / np.max(y)
    peak = np.argmax(y)
    p10, p50, p90 = np.interp([0.1,0.5,0.9], y[:peak+1], x[:peak+1])
    right = np.where(y[peak:] <= .5)[0][0] + peak
    q50 = np.interp(.5, y[right-1:right+1][::-1], x[right-1:right+1][::-1])
    last = np.where(np.abs(y) >= .01)[0][-1]
    p01 = np.interp(.01, np.abs(y[last:last+2])[::-1], x[last:last+2][::-1])
    return dict(peak_ns=float(x[peak]), peak_value=float(np.max(y)), rise_ns=float(p90-p10),
                fwhm_ns=float(q50-p50), half_crossings_ns=[float(p50),float(q50)],
                last_1pct_ns=float(p01), undershoot_pct=float(100*y.min()),
                undershoot_ns=float(x[y.argmin()]))

d = np.arange(100001)*.001
raw = sipm.pulse(d*NS)
# Reuse the chain's TIA transfer with an extended quiet diagnostic time axis
# so that the 1% settling criterion includes the SiPM recharge tail.
tia = filter_analog(raw * sipm.q_pe, 1e12, chain.tia.transimpedance, chain.tia.poles, chain.tia.zeros)
raw_info, tia_info = shape_metrics(d,raw), shape_metrics(d,tia)
tr = np.load(data_dir / 'trials.npz')
truth = tr['true_arrival_s']
est = tr['estimated_arrival_s']
errors = np.abs(est - truth[None,:,:,None])*1e12
count = np.count_nonzero(errors[0,0,:,1] <= 17.6)

ta = t / NS
early = waveform(21)
late = waveform(24.7)
quiet_adc = np.sort(errors[0, 0, :, 1])
print(json.dumps({'sipm_shape': raw_info, 'tia_shape': tia_info,
                  'quiet_adc_within_17_6_ps': int(count)}, indent=2))

blue, orange, grey = '#125F83', '#A74721', '#596879'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#263649','xtick.color':'#263649','ytick.color':'#263649',
                     'font.family':'DejaVu Sans','savefig.dpi':210})

def save(fig, stem):
    fig.savefig(figure_dir / (stem + '.png'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, (ax,tail) = plt.subplots(1,2,figsize=(10.6,4.0),constrained_layout=True)
for axis in (ax,tail):
    axis.axhline(0,color=grey,lw=.8)
    axis.plot(d,raw/raw.max(),c=blue,lw=2,label='SiPM current')
    axis.plot(d,tia/tia.max(),c=orange,lw=2,label='TIA voltage')
    axis.set_xlabel('Delay after arrival (ns)')
ax.axhline(.5,color=grey,lw=1,ls=':')
ax.set(xlim=(0,4),ylim=(-.27,1.08),ylabel='Response / positive peak',title='Positive peak and undershoot')
ax.legend(frameon=False)
tail.axhline(-.01,color=grey,lw=1,ls=':',label='1% of positive peak')
tail.set(xlim=(3,50),ylim=(-.115,.025),title='Recharge tail')
tail.legend(frameon=False,loc='lower right',fontsize=9)
save(fig,'single_pe_template')

fig,ax=plt.subplots(figsize=(9.4,4.5),constrained_layout=True)
ax.axvspan(25,27,color='#e8edf1',label='Outside the readout window')
for arrival,v,color in ((21,early,blue),(24.7,late,orange)):
    ax.plot(ta,v*1e3,c=color,lw=2,label=f'Arrival at {arrival:g} ns')
ax.axvline(25,color=grey,ls='--',lw=1.2)
ax.set(xlim=(20,26.95),xlabel='Time in event (ns)',ylabel='TIA output (mV)',
       title='The 25 ns readout boundary truncates the late pulse')
ax.legend(frameon=False,loc='upper center',fontsize=10)
save(fig,'readout_window_cutoff')

fig,(ax,late_ax)=plt.subplots(1,2,figsize=(12.6,4.65),constrained_layout=True)
for k,label,ls in ((0,'No added noise','-'),(1,'Default noise','--')):
    for n,stage,c in ((0,'TIA',blue),(1,'ADC',orange)):
        e=np.sort(errors[k,0,:,n])
        ax.plot(e,np.arange(1,1+e.size)/e.size*100,ls=ls,color=c,lw=2,label=f'{stage}, {label.lower()}')
ax.axhline(95,color=grey,ls=':',lw=1.1)
ax.scatter([np.percentile(quiet_adc,95)],[95],color=orange,s=31,zorder=4)
ax.annotate('17.6 ps',xy=(np.percentile(quiet_adc,95),95),xytext=(7,73),
            arrowprops={'arrowstyle':'-','color':orange},color=orange)
ax.set(xscale='log',xlim=(.0005,30000),ylim=(0,102),xlabel='Absolute timing error (ps)',
       ylabel='Events at or below error (%)',title='Arrivals within calibration range')
ax.legend(frameon=False,fontsize=9,loc='lower right')
for n,stage,c in ((0,'TIA',blue),(1,'ADC',orange)):
    late_ax.scatter(truth[1]/NS,errors[0,1,:,n],c=c,s=11,alpha=.5,label=stage)
late_ax.axhline(100,ls=':',lw=1.1,c=grey)
late_ax.set(yscale='log',xlim=(20,25),xlabel='True arrival (ns)',ylabel='Absolute timing error (ps)',
            title='Late arrivals, no added noise')
late_ax.legend(frameon=False,loc='lower right')
save(fig,'single_pe_timing_errors')
