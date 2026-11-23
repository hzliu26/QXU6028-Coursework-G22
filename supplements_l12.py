"""Course L2 gate: check the ideal-cone wings separately from the sample fit."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullLocator
import numpy as np

from analysis import KB
from figures import BLUE, RED, GREY, panel, save, setup


def write_l2_gate(out, diag, arrays):
    """Save the cone-wing diagnostic and signed crossing evidence for this dataset."""
    out = Path(out)
    tables = out / 'tables'
    figures = out / 'figures'
    tables.mkdir(exist_ok=True, parents=True)
    figures.mkdir(exist_ok=True, parents=True)
    l2 = arrays[1]
    meta = diag['L2']
    z = l2['ef'] / (KB * meta['T_K'])
    hall, sheet = l2['curves'][0].T
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.0), layout='constrained')
    wing_records = []
    for sign, name, color, marker in ((-1, 'Negative', BLUE, 'o'), (1, 'Positive', RED, '^')):
        mask = (sign * z >= 6 - 1e-10) & (sign * z <= 8 + 1e-10)
        x, y = sheet[mask], np.abs(hall[mask])
        slope = np.polyfit(np.log(x), np.log(y), 1)[0]
        wing_records.append([name, 6, 8, int(np.sum(mask)), slope])
        axes[0].scatter(x, y, s=12, marker=marker, facecolors='none',
                        edgecolors=color, alpha=.8, lw=.65,
                        label=f'{name} wing   slope {slope:.2f}')
    # Anchor a slope-two guide to the geometric centre of the displayed data.
    mask = (np.abs(z) >= 6 - 1e-10) & (np.abs(z) <= 8 + 1e-10)
    x = np.array([sheet[mask].min(), sheet[mask].max()])
    scale = np.exp(np.mean(np.log(np.abs(hall[mask])) - 2 * np.log(sheet[mask])))
    axes[0].plot(x, scale * x**2, ls=':', lw=1.15, color=GREY, label='Slope 2 reference')
    axes[0].set(xscale='log', yscale='log',
                xlabel=r'Normalised $R_s$', ylabel=r'Normalised $|R_H|$')
    for axis in (axes[0].xaxis, axes[0].yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_major_formatter(FormatStrFormatter('%.2f'))
        axis.set_minor_locator(NullLocator())
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5,-.29), fontsize=6.8)
    panel(axes[0], 'a', 'Ideal-cone wings, 6–8 thermal energies')

    ids = np.asarray(l2['ids'])
    repeats = np.asarray(l2['reps'])
    # Use signed raw measurements and one common scale so that zero stays visible.
    raw_scale = np.max(np.abs(repeats[:, :, 0]))
    ax = axes[1]
    for k in range(repeats.shape[1]):
        ax.scatter(ids, repeats[:, k, 0] / raw_scale, s=6, color=BLUE,
                   alpha=.4, label='Individual repeats' if k == 0 else None)
    ax.plot(ids, repeats[:, :, 0].mean(axis=1) / raw_scale,
            color=BLUE, lw=.8, label='Repeat mean')
    ax.axhline(0, color=GREY, lw=.7)
    for left, right in meta['zero_crossing_brackets']:
        ax.axvspan(left, right, color=RED, alpha=.19, lw=0)
    brackets = ', '.join(f'{a}–{b}' for a,b in meta['zero_crossing_brackets'])
    ax.set(xlabel='Measurement index', ylabel=r'Signed $R_H$ / max $|R_H|$ of repeats')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.29),fontsize=6.8)
    panel(ax, 'b', f'Measured Hall crossing at {brackets}')
    save(fig, figures, 'L2_gate_cone_wings_and_neutrality')
    with (tables / 'L2_ideal_cone_wing_slopes.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('EF_sign', 'min_abs_EF_over_kBT', 'max_abs_EF_over_kBT',
                         'grid_points', 'slope_log_abs_RH_vs_log_Rs'))
        writer.writerows(wing_records)
