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
"""The phase diagram of collapse without rotation (Section 4 of paper/collapse-without-rotation.tex). NUMERICAL:
binary64, not interval arithmetic.

Input: data/phase-diagram/alpha-star-phase-diagram.csv (the table of Figure 1) and the stored configurations of
every row (data/phase-diagram/*.json). For every row the program
  1. finds its stored configuration and checks, independently of the gauge, that it is a self-similar collapse
     without rotation at alpha = alpha*: kappa from a least-squares fit of V = kappa z + c over all vortices,
     relative residual, P = |Im kappa|/(-2 Re kappa) = 0, all circulations and their sum nonzero, vortices distinct;
     and that the closest pair and circulation span agree with the table;
  2. checks the first-order condition of a local minimum of alpha on the set Z of collapses without rotation
     (the gradient of alpha lies in the row space of the constraint Jacobian) and computes the eigenvalues of the
     reduced Hessian of the Lagrangian (positive definite: a strict local minimum of alpha on Z);
  3. measures the C2 defect (z -> -z with equal circulations): zero for the odd Euler branch, not for the others.
Then, from the table: the least N with a collapse without rotation for alpha = 6, 5, 4, 3, 2.5, 2, 1.5, 1.2, 1; the
alpha = 2 values for N = 9, 10, 11; the quantity N^2 |d alpha*/dN| (the '11/N law', which fails); the local exponent
p_eff; the fits of alpha_inf over forms and windows (written to data/phase-diagram/alpha-inf-fits.json), and the
fits that force alpha_inf = 0. Finally the stored end points of the searches of Section 4.2 (data/searches/) are
rechecked, and the four-vortex continuation, which fails to give a clean threshold, is described.

Needs numpy and scipy (code/requirements.txt). Run: python3 code/verify_phase_diagram.py. Prints every check and
exits with status 1 if any fails; output in data/verify-phase-diagram.txt. About a minute.
"""
import os
import sys
import csv
import json
import time
import warnings
import numpy as np
from scipy.optimize import curve_fit, linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collapse_core import vel, pack, unpack, System, jac_full, normalize, gauge, rescale_G1, free_mask, null_basis, TWO_PI  # noqa: E402

warnings.filterwarnings('ignore')
DATA = os.path.join(os.path.dirname(HERE), 'data')
PD = os.path.join(DATA, 'phase-diagram')
SE = os.path.join(DATA, 'searches')
OUT = []
FAILED = []
NCHK = [0]
T0 = time.time()


def say(s=''):
    print(s)
    sys.stdout.flush()
    OUT.append(s)


def check(name, ok, detail=''):
    NCHK[0] += 1
    say(('OK    ' if ok else 'FAIL  ') + name + (('   [' + detail + ']') if detail else ''))
    if not ok:
        FAILED.append(name)


def cfg(t):
    return np.array([complex(*p) for p in t['z']]), np.array(t['G'], float)


def collapse_check(z, G, a):
    """Gauge-free: V = kappa z + c by least squares over all vortices. Returns relative residual, P, Re kappa."""
    V = vel(z, G, a)
    A = np.stack([z, np.ones_like(z)], 1)
    (k, c), *_ = np.linalg.lstsq(A, V, rcond=None)
    res = np.max(np.abs(V - k*z - c))/np.max(np.abs(V))
    return res, abs(k.imag)/(-2*k.real), k.real


def kkt_state(z, G, a, h=1e-6):
    """First-order residual and reduced-Hessian eigenvalues of min alpha on Z at a stored point (no iteration)."""
    N = len(z)
    z, G, b = normalize(z, G, a)
    z, G, ir, ig = gauge(z, G)
    z, G = rescale_G1(z, G, a, ig)
    u = pack(z, G, a, 0.0)
    S = System(u, N, free_mask(N, ir, ig, True, False))
    v = u[S.idx]
    J = S.J(v)
    ia = list(S.idx).index(3*N)
    g = np.zeros(len(v))
    g[ia] = 1.0
    lam = np.linalg.lstsq(J.T, g, rcond=None)[0]
    first = np.linalg.norm(g - J.T @ lam)
    n = len(v)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h*max(1.0, abs(v[i]))
        H[:, i] = -(S.J(v + e).T @ lam - S.J(v - e).T @ lam)/(2*e[i])
    H = 0.5*(H + H.T)
    Zb, s = null_basis(J)
    ev = np.linalg.eigvalsh(Zb.T @ H @ Zb)
    return first, ev, np.linalg.norm(S.F(v))


def c2_defect(z, G):
    C = np.abs(z[:, None] + z[None, :])/np.max(abs(z)) + np.abs(G[:, None] - G[None, :])/np.max(abs(G))
    r, c = linear_sum_assignment(C)
    return C[r, c].max()


say('Phase diagram of collapse without rotation: alpha*(N) along four families (NUMERICAL, binary64)')
rows = list(csv.DictReader(open(os.path.join(PD, 'alpha-star-phase-diagram.csv'))))
say('%d rows: %s' % (len(rows), ', '.join('%s %d' % (f, sum(1 for r in rows if r['family'] == f)) for f in ('S', 'E-odd(C2)', 'E-even', 'E-asym'))))

store = {}
for t in json.load(open(os.path.join(PD, 'thresholds-recorded-polished.json'))):
    fam = 'S' if t['family_alpha'] == 1 else ('E-odd(C2)' if t['N'] % 2 else 'E-even')
    store.setdefault((fam, t['N']), t)
for t in json.load(open(os.path.join(PD, 'thresholds-small-n.json'))):
    if t['minimum_index'] == 0:
        store.setdefault(('S', t['N']), t)
for f in ('family-s-n60.json', 'family-s-n61-64.json', 'family-s-n65-73.json'):
    for t in json.load(open(os.path.join(PD, f))):
        store.setdefault(('S', t['N']), t)
for t in json.load(open(os.path.join(PD, 'family-e-odd-n31-79.json'))):
    store.setdefault(('E-odd(C2)', t['N']), t)
for f in ('family-e-asym-n31-37.json', 'family-e-asym-n38.json'):
    for t in json.load(open(os.path.join(PD, f))):
        store.setdefault(('E-asym', t['N']), t)

say('\n1-3. Every row against its stored configuration')
worst = dict(res=0.0, P=0.0, dA=0.0, first=0.0, pair=0.0, span=0.0)
nposdef = {}
defects = {}
missing = []
for r in rows:
    key = (r['family'], int(r['N']))
    if key not in store:
        missing.append(key)
        continue
    t = store[key]
    z, G = cfg(t)
    a = float(t['alpha_star'])
    res, P, rek = collapse_check(z, G, a)
    worst['res'] = max(worst['res'], res)
    worst['P'] = max(worst['P'], P)
    worst['dA'] = max(worst['dA'], abs(a - float(r['alpha_star'])))
    d = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(d, np.inf)
    pair = d.min()/np.max(np.abs(z))
    span = np.abs(G).max()/np.abs(G).min()
    worst['pair'] = max(worst['pair'], abs(pair/float(r['closest_pair_over_size']) - 1))
    worst['span'] = max(worst['span'], abs(span/float(r['circulation_span']) - 1))
    ok_basic = rek < 0 and np.all(G != 0) and abs(G.sum()) > 1e-3*np.abs(G).sum() and pair > 1e-3
    if not ok_basic:
        check('%s N=%d: collapse with nonzero circulations and total, distinct vortices' % key, False)
    first, ev, fres = kkt_state(z, G, a)
    worst['first'] = max(worst['first'], first)
    nposdef.setdefault(r['family'], [0, 0, []])
    nposdef[r['family']][1] += 1
    if ev.min() > 0:
        nposdef[r['family']][0] += 1
    else:
        nposdef[r['family']][2].append(int(r['N']))
    defects.setdefault(r['family'], []).append((int(r['N']), c2_defect(z, G)))
say('   (%.0f s)' % (time.time() - T0))
check('every row has a stored configuration', not missing, str(missing) if missing else '')
check('every stored configuration is a self-similar collapse at its alpha*: largest relative residual %.1e < 1e-9' % worst['res'], worst['res'] < 1e-9)
check('without rotation: largest P = |Im kappa|/(-2 Re kappa) is %.1e < 1e-9' % worst['P'], worst['P'] < 1e-9)
check('alpha* stored agrees with the table (largest difference %.1e)' % worst['dA'], worst['dA'] < 1e-9)
br = [r for r in rows if r['recorded_bracket_lo']]
check('the %d refined thresholds of the recorded families lie inside their recorded brackets of width 1e-4 or less' % len(br),
      all(float(r['recorded_bracket_lo']) - 1e-9 <= float(r['alpha_star']) <= float(r['recorded_bracket_hi']) + 1e-9 for r in br)
      and max(float(r['recorded_bracket_hi']) - float(r['recorded_bracket_lo']) for r in br) <= 1.0001e-4)
check('closest pair and circulation span agree with the table to 4 digits (largest relative differences %.1e, %.1e)' % (worst['pair'], worst['span']),
      worst['pair'] < 1e-3 and worst['span'] < 1e-3)
check('first-order condition of min alpha on Z at every row: largest residual %.1e < 1e-7' % worst['first'], worst['first'] < 1e-7)
for f in ('S', 'E-even', 'E-asym'):
    npd, tot, bad = nposdef[f]
    check('family %s: reduced Hessian positive definite (a strict local minimum of alpha on Z) at %d of %d rows' % (f, npd, tot), npd == tot,
          'not at N = %s' % bad if bad else '')
npd, tot, bad = nposdef['E-odd(C2)']
say('   family E-odd(C2): reduced Hessian in the full space positive definite at %d of %d rows%s' % (npd, tot, (' (indefinite at N = %s)' % bad) if bad else ''))
check('family E-odd(C2), N = 13..29 (recorded): strict local minima of alpha on Z in the full space', not [N for N in bad if N <= 29])
dodd = [x for N, x in defects['E-odd(C2)']]
deven = [x for N, x in defects['E-even']]
dS = [x for N, x in defects['S']]
check('the odd Euler branch is exactly C2-symmetric (a central vortex and pairs z, -z with equal circulations): largest defect %.1e' % max(dodd), max(dodd) < 1e-8)
check('the even Euler branch and the SQG-type family are not C2-symmetric: least defects %.2f and %.2f' % (min(deven), min(dS)), min(deven) > 1e-2 and min(dS) > 1e-2)

say('\n4. Least N with a collapse without rotation, along these families  (%.0f s)' % (time.time() - T0))
byN = {}
for r in rows:
    N = int(r['N'])
    byN[N] = min(byN.get(N, 99.0), float(r['alpha_star']))
Nall = sorted(byN)
want = {6: 5, 5: 6, 4: 6, 3: 8, 2.5: 9, 2: 11, 1.5: 17, 1.2: 29, 1: 60}
got = {a: next(N for N in Nall if byN[N] <= a) for a in want}
for a in want:
    check('alpha = %g: least N = %d' % (a, got[a]), got[a] == want[a])
fams = {}
for r in rows:
    fams.setdefault(r['family'], {})[int(r['N'])] = float(r['alpha_star'])
S = fams['S']
check('S is the lowest family at every N where another exists', all(S[N] <= min(fams[f][N] for f in fams if N in fams[f]) for N in S))
check('alpha = 2: alpha*(9) = %.10f > 2, alpha*(10) = %.10f > 2, alpha*(11) = %.10f < 2' % (S[9], S[10], S[11]),
      S[9] > 2 and S[10] > 2 and S[11] < 2 and abs(S[10] - 2.0913751011) < 1e-9)
E = fams['E-odd(C2)']
check('SQG (alpha = 1): the S family crosses between N = 59 (%.5f) and 60 (%.5f); the odd Euler branch between N = 73 (%.5f) and 75 (%.5f)' %
      (S[59], S[60], E[73], E[75]), S[59] > 1 > S[60] and E[73] > 1 > E[75])
check('alpha*(N) decreases along S (N = 5..73) and along the odd Euler branch (N = 13..79)',
      all(S[N + 1] < S[N] for N in range(5, 73)) and all(E[N + 2] < E[N] for N in range(13, 79, 2)))
check('least values computed: %.5f (S, N = 73) and %.5f (odd Euler, N = 79), both positive' % (S[73], E[79]), S[73] > 0 and E[79] > 0)

say('\n5. The 11/N law does not hold; the local exponent drifts')
dd = [(n, (S[n] - S[n + 1])*n*(n + 1)) for n in range(5, 73)]
nmin, vmin = min(dd[20:], key=lambda x: x[1])
say('   N^2 |d alpha*/dN| along S: ' + ' '.join('%d:%.2f' % x for x in dd[::5]))
check('N^2 |d alpha*/dN| along S has its least value %.2f at N = %d and then rises to %.2f at N = 72 (not constant)' % (vmin, nmin, dd[-1][1]),
      30 <= nmin <= 40 and dd[-1][1] > vmin + 0.5)
ddE = [(n, (E[n] - E[n + 2])/2*n*(n + 2)) for n in range(13, 79, 2)]
nminE, vminE = min(ddE, key=lambda x: x[1])
say('   odd Euler branch: ' + ' '.join('%d:%.2f' % x for x in ddE[::4]))
check('along the odd Euler branch the same quantity stops falling at %.2f near N = %d and turns up' % (vminE, nminE), 60 <= nminE <= 75 and ddE[-1][1] > vminE)
peff = []
for i in range(1, len(dd) - 1):
    n0, v0 = dd[i - 1]
    n1, v1 = dd[i + 1]
    peff.append((dd[i][0], 1 - np.log(v1/v0)/np.log(n1/n0)))
pe = dict(peff)
check('local exponent p_eff (alpha* ~ a + b N^-p) along S falls steadily: %.2f (N = 30), %.2f (N = 48), %.2f (N = 66)' % (pe[30], pe[48], pe[66]),
      pe[30] > pe[48] > pe[66] and pe[66] < 0.9)

say('\n6. Fits of alpha_inf over forms and windows  (%.0f s)' % (time.time() - T0))
forms = {
    'a+b/N+c/N^2': (lambda N, a, b, c: a + b/N + c/N**2, [1, 10, 0]),
    'a+b/N+c/N^2+d/N^3': (lambda N, a, b, c, d: a + b/N + c/N**2 + d/N**3, [1, 10, 0, 0]),
    'a+b*N^-p': (lambda N, a, b, p: a + b*N**(-p), [1, 10, 1]),
    'a+(b ln N + c)/N': (lambda N, a, b, c: a + (b*np.log(N) + c)/N, [1, 1, 1]),
    'a+b/(ln N)^q': (lambda N, a, b, q: a + b/np.log(N)**q, [0, 10, 2]),
    'b/(ln N)^q [alpha_inf = 0 forced]': (lambda N, b, q: b/np.log(N)**q, [5, 1]),
    'b*N^-p [alpha_inf = 0 forced]': (lambda N, b, p: b*N**(-p), [5, 0.3]),
}
fits_out = {}
for f, d in (('S', S), ('E-odd(C2)', E)):
    Ns = sorted(d)
    span = Ns[-1] - Ns[0]
    res = []
    for frac in (0.25, 0.4, 0.6):
        lo = Ns[-1] - int(frac*span)
        Nv = np.array([n for n in Ns if n >= lo], float)
        yv = np.array([d[int(n)] for n in Nv])
        for name, (fn, p0) in forms.items():
            p, _ = curve_fit(fn, Nv, yv, p0=p0, maxfev=40000)
            rms = float(np.sqrt(np.mean((fn(Nv, *p) - yv)**2)))
            res.append(dict(Nmin=lo, form=name, alpha_inf=0.0 if 'forced' in name else float(p[0]), rms=rms, npts=len(Nv)))
            say('   %-10s N >= %2d (%2d pts) %-36s alpha_inf %.4f  rms %.1e' % (f, lo, len(Nv), name, res[-1]['alpha_inf'], rms))
    free = [x for x in res if 'forced' not in x['form']]
    forced = [x for x in res if 'forced' in x['form']]
    lo_a, hi_a = min(x['alpha_inf'] for x in free), max(x['alpha_inf'] for x in free)
    same = []
    best = []
    for x in forced:
        twin = 'a+b/(ln N)^q' if 'ln N' in x['form'] else 'a+b*N^-p'
        tw = next(y for y in free if y['form'] == twin and y['Nmin'] == x['Nmin'])
        same.append(x['rms']/tw['rms'])
        best.append(x['rms']/min(y['rms'] for y in free if y['Nmin'] == x['Nmin']))
    fits_out[f] = dict(alpha_inf_range=[lo_a, hi_a], forced_over_same_form=[min(same), max(same)], forced_over_best=[min(best), max(best)], fits=res)
    check('%s: every free fit gives alpha_inf in [%.3f, %.3f], inside [0.64, 0.81]' % (f, lo_a, hi_a), 0.64 <= lo_a and hi_a <= 0.81)
    check('%s: fits forcing alpha_inf = 0 have rms %.0f to %.0f times that of the same form with alpha_inf free, and %.0f to %.0f times the best free fit' %
          (f, min(same), max(same), min(best), max(best)), min(same) > 20)
json.dump(fits_out, open(os.path.join(PD, 'alpha-inf-fits.json'), 'w'), indent=1)

say('\n7. The searches of Section 4.2: stored end points  (%.0f s)' % (time.time() - T0))
for f, N, a0 in (('random-walk-n9.json', 9, 2.3000208434), ('random-walk-n10.json', 10, 2.0913751011)):
    recs = json.load(open(os.path.join(SE, f)))
    ok = True
    for t in recs:
        z, G = cfg(t)
        res, P, rek = collapse_check(z, G, t['alpha'])
        ok = ok and res < 1e-9 and P < 1e-9 and rek < 0
    check('random walk on Z, N = %d: %d distinct local minima of alpha (%s), landings %s; each a collapse without rotation; least %.10f' %
          (N, len(recs), ', '.join('%.10f' % t['alpha'] for t in recs), '+'.join(str(t['count']) for t in recs), min(t['alpha'] for t in recs)),
          ok and abs(min(t['alpha'] for t in recs) - a0) < 1e-9)
for f, N, P0 in (('p-search-alpha2-n9.json', 9, 0.0675758057), ('p-search-alpha2-n10.json', 10, 0.0237078013)):
    recs = json.load(open(os.path.join(SE, f)))
    ok = True
    for t in recs:
        z, G = cfg(t)
        res, P, rek = collapse_check(z, G, 2.0)
        ok = ok and res < 1e-9 and abs(P - t['P']) < 1e-8 and rek < 0
    check('minimization of P at alpha = 2, N = %d: local minima %s, least %.10f > 0; each a collapse at alpha = 2 with that P' %
          (N, ', '.join('%.10f' % t['P'] for t in recs), min(t['P'] for t in recs)), ok and abs(min(t['P'] for t in recs) - P0) < 1e-9)
ms = json.load(open(os.path.join(SE, 'continuation-from-minima.json')))
least = {N: min(t['alpha_star'] for t in ms if t['N'] == N) for N in (8, 9, 10)}
check('continuation in alpha of the %d recorded P-minima for N = 8, 9, 10 (alpha = 0, 1, 2) up to P = 0, then min alpha on Z: end values %s; '
      'the least for each N is the S threshold' % (len(ms), ', '.join(sorted(set('%.10f' % t['alpha_star'] for t in ms)))),
      all(abs(least[N] - S[N]) < 1e-8 for N in least))
rm = json.load(open(os.path.join(SE, 'removal-alpha2-n11.json')))
ends = [t['alpha_end'] for t in rm]
okr = True
for t in rm:
    u = np.array(t['u'])
    z, G, a, b = unpack(u, 11)
    keep = [j for j in range(11) if j != t['k']]
    res, P, rek = collapse_check(z[keep], G[keep], a)
    okr = okr and res < 1e-8 and P < 1e-8 and abs(G[t['k']]) < 1e-12
check('vortex removal from the alpha = 2, N = 11 collapse: all %d weak vortices lead to alpha* = %.10f (the N = 10 threshold); '
      'the remaining ten form a collapse without rotation there' % (len(rm), np.mean(ends)), okr and max(abs(e - S[10]) for e in ends) < 1e-9)
sym = json.load(open(os.path.join(SE, 'c2-symmetric-thresholds.json')))
for t in sym:
    z, G = cfg(t)
    res, P, rek = collapse_check(z, G, t['alpha_star'])
    check('C2-symmetric reduced system (%s%d pairs): N = %d, least alpha found %.10f; a collapse without rotation (residual %.1e), C2 defect %.1e' %
          ('centre + ' if t['centre'] else '', t['rings'], t['N'], t['alpha_star'], res, c2_defect(z, G)),
          res < 1e-9 and P < 1e-9 and c2_defect(z, G) < 1e-8)
t4 = json.load(open(os.path.join(SE, 'four-vortex-continuation.json')))
z, G = cfg(t4)
res, P, rek = collapse_check(z, G, t4['alpha_star'])
gs = sorted(np.abs(G))
say('   four vortices (FAILED to give a clean threshold): P reaches 0 near alpha = %.2f (residual %.1e), where the three weaker circulations '
    'are %.1e, %.1e, %.1e times the strongest' % (t4['alpha_star'], res, gs[0]/gs[3], gs[1]/gs[3], gs[2]/gs[3]))
check('four vortices: at the end of the continuation three circulations are below 1e-5 of the fourth (a strong vortex with near-tracers)',
      gs[2]/gs[3] < 1e-5 and t4['alpha_star'] > 14)

say('\n%d checks, %d failed  (%.0f s)' % (NCHK[0], len(FAILED), time.time() - T0))
with open(os.path.join(DATA, 'verify-phase-diagram.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
