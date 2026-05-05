#%% Imports
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from scipy.signal.windows import blackman
import matplotlib.pyplot as plt
import os
import pandas as pd

#%% Parameters
main_dir     = "/Volumes/tungsten/scratch/gfelsenb/Hanna/2p-imaging"
container_id = 'PPL101'
data_file    = 'entire_trace_with_odors_new.csv'

#%% Helper: compute amplitude spectrum
def compute_spectrum(segment, fs):
    seg_detrended = detrend(segment)
    n = len(seg_detrended)
    w = blackman(n)
    yf = rfft(seg_detrended * w)
    xf = rfftfreq(n, d=1/fs)
    amp = (2.0 / np.sum(w)) * np.abs(yf)
    dom_idx = np.argmax(amp)
    return xf, amp, xf[dom_idx], amp[dom_idx]

#%% Batch processing
container_dir = os.path.join(main_dir, container_id)
freq_data_dir = os.path.join(container_dir, 'processed frequency data')
os.makedirs(freq_data_dir, exist_ok=True)

day_dirs = sorted([
    d for d in os.listdir(container_dir)
    if os.path.isdir(os.path.join(container_dir, d)) and d != 'processed frequency data'
])

for day_id in day_dirs:
    day_path = os.path.join(container_dir, day_id)

    # Load logfile for this day
    logfile_path = os.path.join(day_path, f'{day_id}_logfile.csv')
    if not os.path.exists(logfile_path):
        print(f"[{day_id}] Logfile not found, skipping day.")
        continue
    logfile = pd.read_csv(logfile_path)

    fly_dirs = sorted([
        f for f in os.listdir(day_path)
        if os.path.isdir(os.path.join(day_path, f)) and f.startswith('fly')
    ])

    for fly_id in fly_dirs:
        data_path = os.path.join(day_path, fly_id, 'session_post', data_file)
        if not os.path.exists(data_path):
            print(f"[{day_id}/{fly_id}] Data file not found, skipping.")
            continue

        # Extract fly number and look up conditions from logfile
        try:
            fly_num = int(''.join(filter(str.isdigit, fly_id)))
        except ValueError:
            print(f"[{day_id}/{fly_id}] Could not parse fly number, skipping.")
            continue

        log_row = logfile[logfile['fly-num'] == fly_num]
        if log_row.empty:
            print(f"[{day_id}/{fly_id}] No logfile entry for fly-num={fly_num}, skipping.")
            continue

        condition_feeding = str(log_row['starved/refed'].values[0])
        condition_training = str(log_row['mock/trained'].values[0])
        print(f"[{day_id}/{fly_id}] {condition_feeding} | {condition_training}")

        # Load traces
        df = pd.read_csv(data_path)
        time_trace    = df['time'].to_numpy()
        calcium_trace = df['calcium'].to_numpy()
        odor_trace    = df['odor'].to_numpy()
        shock_trace   = df['shock'].to_numpy()

        # Sampling frequency
        time_diffs = np.diff(time_trace)
        if not np.allclose(time_diffs, time_diffs[0]):
            print(f"[{day_id}/{fly_id}] Non-uniform sampling, skipping.")
            continue
        fs = 1 / time_diffs[0]

        # Odor onsets/offsets
        odor_onsets  = np.where(np.diff(odor_trace) == 1)[0] + 1
        odor_offsets = np.where(np.diff(odor_trace) == -1)[0] + 1

        # Compute spectrum
        xf_full, amp_full, dom_freq_full, dom_amp_full = compute_spectrum(calcium_trace, fs)
        print(f"  Dominant Frequency: {dom_freq_full:.4f} Hz")

        # --- Plot ---
        trace_detrended_full = detrend(calcium_trace)
        fig_full, (ax1_full, ax2_full) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        ax1_full.plot(time_trace, calcium_trace, color='gray', alpha=0.4, label='Raw')
        ax1_full.plot(time_trace, trace_detrended_full, color='black', label='Detrended')

        _labeled = False
        for onset, offset in zip(odor_onsets, odor_offsets):
            _label = 'Stimulus' if not _labeled else '_nolegend_'
            ax1_full.axvspan(time_trace[onset], time_trace[offset - 1],
                             color='steelblue', alpha=0.25, label=_label)
            _labeled = True

        ax1_full.set_title(
            f"Full Trace – {day_id} {fly_id} | {condition_feeding} | {condition_training}",
            fontsize=13)
        ax1_full.set_ylabel(r"$\Delta F/F$")
        ax1_full.set_xlabel("Time (s)")
        ax1_full.legend(loc='upper right')
        ax1_full.grid(True, alpha=0.2)

        ax2_full.plot(xf_full, amp_full, color='black', linewidth=1.5)
        ax2_full.axvline(dom_freq_full, color='crimson', linestyle='--', linewidth=1.2,
                         label=f'Dominant: {dom_freq_full:.4f} Hz')
        ax2_full.set_title("Amplitude Spectrum", fontsize=13)
        ax2_full.set_ylabel(r"Amplitude ($\Delta F/F$)")
        ax2_full.set_xlabel("Frequency (Hz)")
        ax2_full.set_xlim(0, fs / 2)
        ax2_full.set_ylim(0, np.max(amp_full) * 1.1)
        ax2_full.legend(loc='upper right', fontsize=9)
        ax2_full.grid(True, alpha=0.2)

        plt.tight_layout()

        # Save figure
        fig_save_dir = os.path.join(day_path, fly_id, 'session_post')
        fig_path = os.path.join(fig_save_dir, f'frequency_analysis_{day_id}_{fly_id}.pdf')
        fig_full.savefig(fig_path, bbox_inches='tight')
        plt.close(fig_full)

        # Save amplitude spectrum with metadata
        npy_path = os.path.join(
            freq_data_dir,
            f'frequency_analysis_{day_id}_{fly_id}_amplitude_spectrum.npy'
        )
        np.save(npy_path, {
            'frequencies':        xf_full,
            'amplitude':          amp_full,
            'dominant_frequency': dom_freq_full,
            'day_id':             day_id,
            'fly_id':             fly_id,
            'fly_num':            fly_num,
            'starved_refed':      condition_feeding,
            'mock_trained':       condition_training,
            'time_trace':         time_trace,
            'calcium_trace':      calcium_trace,
            'odor_onsets':        odor_onsets,
            'odor_offsets':       odor_offsets,
            'fs':                 fs,
        })
        print(f"  Figure saved:   {fig_path}")
        print(f"  Spectrum saved: {npy_path}")

print("Batch processing complete.")

# %%
# =============================================================================
#  POST-BATCH ANALYSIS: load all processed spectra and plot by condition
# =============================================================================

#%% Load all saved amplitude spectra
npy_files = sorted([
    f for f in os.listdir(freq_data_dir)
    if f.endswith('_amplitude_spectrum.npy')
])

records = []
_missing_trace_fields = False
for fname in npy_files:
    data = np.load(os.path.join(freq_data_dir, fname), allow_pickle=True).item()
    if 'time_trace' not in data:
        _missing_trace_fields = True
    records.append(data)

if not records:
    print("No processed spectra found. Run the batch section first.")
else:
    print(f"Loaded {len(records)} spectra.")
    if _missing_trace_fields:
        print("WARNING: some .npy files are missing trace fields (time_trace, calcium_trace, etc.).")
        print("Re-run the batch processing section to regenerate them with the full data.")

#%% Group by combined condition label
from collections import defaultdict

# Interpolate all spectra onto a common frequency axis (finest resolution = most points)
max_len = max(len(r['frequencies']) for r in records)
ref_record = next(r for r in records if len(r['frequencies']) == max_len)
common_xf = ref_record['frequencies']

def interp_to_common(xf, amp, common_xf):
    return np.interp(common_xf, xf, amp)

# Build condition -> list of per-fly entries
cond_groups = defaultdict(list)
for r in records:
    cond_label = f"{r['starved_refed']} | {r['mock_trained']}"
    amp_interp = interp_to_common(r['frequencies'], r['amplitude'], common_xf)
    cond_groups[cond_label].append({
        'amp':           amp_interp,
        'fly_label':     f"{r['day_id']} {r['fly_id']}",
        'dom_freq':      r['dominant_frequency'],
        'time_trace':    r['time_trace'],
        'calcium_trace': r['calcium_trace'],
        'odor_onsets':   r['odor_onsets'],
        'odor_offsets':  r['odor_offsets'],
        'fs':            r['fs'],
    })

conditions = sorted(cond_groups.keys())

# Assign a distinct color per condition
cmap = plt.get_cmap('tab10')
cond_colors = {cond: cmap(i) for i, cond in enumerate(conditions)}

#%% Figure 1: amplitude spectra (left, 3/5) + dominant frequency (middle, 1/5) + dominant amplitude (right, 1/5)
fig_avg = plt.figure(figsize=(16, 5))
gs = fig_avg.add_gridspec(1, 3, width_ratios=[3, 1, 1], wspace=0.1)
ax_avg  = fig_avg.add_subplot(gs[0])
ax_box  = fig_avg.add_subplot(gs[1])
ax_amp  = fig_avg.add_subplot(gs[2])

for cond, entries in cond_groups.items():
    amps = np.vstack([e['amp'] for e in entries])
    mean_amp = np.mean(amps, axis=0)
    sem_amp  = np.std(amps, axis=0) / np.sqrt(len(entries))
    color = cond_colors[cond]

    ax_avg.plot(common_xf, mean_amp, color=color, linewidth=2.0,
                label=f"{cond} (n={len(entries)})")
    ax_avg.fill_between(common_xf,
                        mean_amp - sem_amp,
                        mean_amp + sem_amp,
                        color=color, alpha=0.2)

ax_avg.set_title("Average Amplitude Spectra by Condition (mean ± SEM)", fontsize=13)
ax_avg.set_ylabel(r"Amplitude ($\Delta F/F$)")
ax_avg.set_xlabel("Frequency (Hz)")
ax_avg.set_xlim(0, common_xf[-1])
ax_avg.legend(loc='upper right', fontsize=9)
ax_avg.grid(True, alpha=0.2)

# Collect per-condition dominant frequency and amplitude at dominant frequency
dom_freq_by_cond = []
dom_amp_by_cond  = []
for cond in conditions:
    entries = cond_groups[cond]
    dom_freq_by_cond.append([e['dom_freq'] for e in entries])
    dom_amp_by_cond.append([
        float(np.interp(e['dom_freq'], common_xf, e['amp']))
        for e in entries
    ])

def _draw_boxplot(ax, data_by_cond, title, ylabel):
    bp = ax.boxplot(data_by_cond, patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=2))
    for patch, cond in zip(bp['boxes'], conditions):
        patch.set_facecolor(cond_colors[cond])
        patch.set_alpha(0.7)
    for i, (cond, vals) in enumerate(zip(conditions, data_by_cond)):
        x = np.random.default_rng(seed=42).uniform(i + 0.82, i + 1.18, len(vals))
        ax.scatter(x, vals, color=cond_colors[cond], zorder=3,
                   edgecolors='black', linewidths=0.5, s=40)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, fontsize=8, rotation=30, ha='right')
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.grid(True, axis='y', alpha=0.2)

_draw_boxplot(ax_box, dom_freq_by_cond,
              "Dominant\nFrequency", "Frequency (Hz)")
_draw_boxplot(ax_amp, dom_amp_by_cond,
              "Amplitude at\nDominant Freq.", r"Amplitude ($\Delta F/F$)")

plt.tight_layout()

avg_fig_path = os.path.join(freq_data_dir, 'averaged_spectra_by_condition.pdf')
fig_avg.savefig(avg_fig_path, bbox_inches='tight')
plt.show()
print(f"Averaged spectra figure saved: {avg_fig_path}")

#%% Figure 2+: one figure per condition, one row per fly (trace left, spectrum right)
for cond in conditions:
    entries = cond_groups[cond]
    color = cond_colors[cond]
    n_flies = len(entries)

    fig_ind, axes = plt.subplots(
        n_flies, 2,
        figsize=(14, 3.5 * n_flies),
        squeeze=False,
        gridspec_kw={'width_ratios': [2, 1]},
    )
    fig_ind.suptitle(f"Individual Flies – {cond} (n={n_flies})", fontsize=14, y=1.01)

    for row, e in enumerate(entries):
        ax_tr  = axes[row, 0]
        ax_sp  = axes[row, 1]

        t          = e.get('time_trace')
        ca         = e.get('calcium_trace')
        onsets     = e.get('odor_onsets')
        offsets    = e.get('odor_offsets')
        fly_fs     = e.get('fs')

        if t is None or ca is None:
            ax_tr.text(0.5, 0.5, 'Trace data not available.\nRe-run batch processing.',
                       ha='center', va='center', transform=ax_tr.transAxes, fontsize=10)
            ax_tr.set_title(f"{e['fly_label']}", fontsize=10)
        else:

        else:
            ca_det = detrend(ca)
            # Trace panel
            ax_tr.plot(t, ca,     color='gray',  alpha=0.4, linewidth=0.8, label='Raw')
            ax_tr.plot(t, ca_det, color='black', linewidth=1.0, label='Detrended')
            _first = True
            for on, off in zip(onsets, offsets):
                ax_tr.axvspan(t[on], t[off - 1], color='steelblue', alpha=0.25,
                              label='Stimulus' if _first else '_nolegend_')
                _first = False
            ax_tr.set_title(f"{e['fly_label']}", fontsize=10)
            ax_tr.set_ylabel(r"$\Delta F/F$", fontsize=9)
            ax_tr.set_xlabel("Time (s)", fontsize=9)
            ax_tr.legend(loc='upper right', fontsize=7)
            ax_tr.grid(True, alpha=0.2)

        # Spectrum panel
        ax_sp.plot(common_xf, e['amp'], color=color, linewidth=1.2)
        ax_sp.axvline(e['dom_freq'], color='crimson', linestyle='--', linewidth=1.0,
                      label=f"Dom: {e['dom_freq']:.4f} Hz")
        ax_sp.set_title("Amplitude Spectrum", fontsize=10)
        ax_sp.set_ylabel(r"Amplitude ($\Delta F/F$)", fontsize=9)
        ax_sp.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_sp.set_xlim(0, common_xf[-1])
        ax_sp.set_ylim(0, np.max(e['amp']) * 1.1)
        ax_sp.legend(loc='upper right', fontsize=7)
        ax_sp.grid(True, alpha=0.2)

    plt.tight_layout()
    safe_cond = cond.replace(' ', '_').replace('/', '-').replace('|', '_')
    ind_fig_path = os.path.join(freq_data_dir, f'individual_spectra_{safe_cond}.pdf')
    fig_ind.savefig(ind_fig_path, bbox_inches='tight')
    plt.show()
    print(f"Individual spectra figure saved: {ind_fig_path}")
# %%

