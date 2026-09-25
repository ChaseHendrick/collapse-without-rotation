#!/usr/bin/env python3
# Copyright 2026 Chase Hendrick
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Figure 1 of paper/collapse-without-rotation.tex: the phase diagram alpha*(N) of self-similar collapse without
rotation, from data/phase-diagram/alpha-star-phase-diagram.csv and the fits in data/phase-diagram/alpha-inf-fits.json
(written by verify_phase_diagram.py).

Writes paper/figures/phase-diagram.pdf and paper/figures/phase-diagram.svg. Needs matplotlib and numpy.
Run: python3 code/plot_phase_diagram.py. A few seconds.
"""
import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PD = os.path.join(ROOT, 'data', 'phase-diagram')
FIG = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'svg.hashsalt': 'phase-diagram', 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
# Categorical slots 1-3 of a colour-blind-checked palette (validated all-pairs); the side branch in neutral grey.
STY = [('S', '#2a78d6', 'o', 'SQG-type family S (no symmetry)'),
       ('E-odd(C2)', '#eb6834', 's', 'Euler-type family, odd N (C$_2$-symmetric)'),
       ('E-even', '#1baf7a', 'D', 'Euler-type family, even N'),
       ('E-asym', '#8a8984', 'x', 'Euler-type, continued without C$_2$')]

rows = list(csv.DictReader(open(os.path.join(PD, 'alpha-star-phase-diagram.csv'))))
fits = json.load(open(os.path.join(PD, 'alpha-inf-fits.json')))
lo = min(fits[f]['alpha_inf_range'][0] for f in fits)
hi = max(fits[f]['alpha_inf_range'][1] for f in fits)

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9))
for fam, col, mk, lab in STY:
    N = np.array([int(r['N']) for r in rows if r['family'] == fam])
    a = np.array([float(r['alpha_star']) for r in rows if r['family'] == fam])
    for k in range(2):
        x = N if k == 0 else 1/N
        if mk == 'x':
            ax[k].plot(x, a, mk, ms=3.6, mew=0.8, color=col, label=lab, zorder=3)
        else:
            ax[k].plot(x, a, mk, ms=3.2 if mk != 'D' else 2.8, mew=0.8, mfc='white', color=col, label=lab, zorder=3)
for k in range(2):
    for yv, txt in ((1, r'SQG, $\alpha = 1$'), (2, r'$\alpha = 2$')):
        ax[k].axhline(yv, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax[k].set_ylabel(r'threshold $\alpha^*(N)$', color=INK)
    ax[k].tick_params(colors=INK, labelsize=7.5)
    for s in ('top', 'right'):
        ax[k].spines[s].set_visible(False)
    ax[k].grid(True, color=GRID, lw=0.4, zorder=0)
ax[0].set_xlabel('number of vortices $N$')
ax[0].set_xlim(0, 82)
ax[0].set_ylim(0, 6.2)
ax[0].text(80, 2.06, r'$\alpha = 2$: $N \geq 11$', ha='right', va='bottom', fontsize=7, color=MUTED)
ax[0].text(80, 1.06, r'SQG: $N \geq 60$', ha='right', va='bottom', fontsize=7, color=MUTED)
ax[0].legend(fontsize=6.5, frameon=False, loc='upper right', handletextpad=0.3, borderaxespad=0.2)
ax[1].set_xlabel('$1/N$')
ax[1].set_xlim(0, 0.08)
ax[1].set_ylim(0, 2.2)
ax[1].axhspan(lo, hi, color='#2a78d6', alpha=0.12, lw=0, zorder=0)
ax[1].text(0.002, (lo + hi)/2, r'fitted $\alpha_\infty$: %.2f to %.2f' % (lo, hi), va='center', fontsize=7, color=INK)
ax[1].text(0.079, 2.03, r'$\alpha = 2$', ha='right', va='bottom', fontsize=7, color=MUTED)
ax[1].text(0.079, 1.03, r'SQG', ha='right', va='bottom', fontsize=7, color=MUTED)
fig.tight_layout(w_pad=2.0)
fig.savefig(os.path.join(FIG, 'phase-diagram.pdf'), metadata={'CreationDate': None, 'ModDate': None})
fig.savefig(os.path.join(FIG, 'phase-diagram.svg'), metadata={'Date': None})
print('wrote paper/figures/phase-diagram.pdf and .svg; alpha_inf band %.3f .. %.3f' % (lo, hi))
