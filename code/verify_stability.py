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
"""Linear stability, and perturbed direct integration, of the two collapses without rotation of the companion paper
(alpha = 2 with eleven vortices, and SQG with sixty), Section 5.3 of paper/collapse-without-rotation.tex. NUMERICAL:
binary64.

Similarity variables z = rho(t) zeta, d tau = rho^(-2 beta) dt, beta = 1 + alpha/2: the collapse is a fixed point of
d zeta/d tau = V(zeta) - kappa zeta. With 2 pi kappa = -1 the linearization (circulations held fixed) is J_E/(2 pi),
E = 2 pi V + zeta, and a mode with exponent k = Re eig(J_E) grows relative to the size r of the configuration like
r^(-k). The linearization of 2 pi V is a Hamiltonian matrix (its eigenvalues pair as mu, -mu), so the exponents pair
as k + k' = 2; the six exponents of the symmetries are 0, 2 beta, 1, 1, 2 and -alpha. Both are checked.

Integration: DOP853, rtol 1e-13, of the unperturbed stored configuration and of three random position perturbations
of relative size 1e-6, to 1.5 t_c (alpha = 2) and 1.2 t_c (SQG). Departure = shape deviation (after the best rotation
and scaling) of 1e-2. Reported: size and rotation at departure, least size in the window, closest pair, and the drift
of the invariants.

Inputs: data/collapse-alpha2-n11-no-rotation.json and data/collapse-sqg-n60-no-rotation.json (the configurations of
the companion paper's Theorem 5). Needs numpy and scipy. Run: python3 code/verify_stability.py [--quick].
The full run takes about two and a half minutes, almost all of it the four SQG integrations; --quick integrates
the SQG runs only to 0.08 t_c, which covers the departure but not the least size (about 20 s). Prints every check and
exits with status 1 if any fails; output in data/verify-stability[-quick].txt and data/stability-*.json.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collapse_core import vel, pack, unpack, System, jac_full, normalize, gauge, rescale_G1, free_mask, newton_project, diagnostics, TWO_PI  # noqa: E402

warnings.filterwarnings('ignore')
DATA = os.path.join(os.path.dirname(HERE), 'data')
QUICK = '--quick' in sys.argv
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


def load(case):
    if case == 'alpha2-n11':
        d = json.load(open(os.path.join(DATA, 'collapse-alpha2-n11-no-rotation.json')))
        m = d['minima'][0]['config']
        z = np.array([complex(*p) for p in m['z']])
        G = np.array(m['G'], float)
        a = 2.0
    else:
        d = json.load(open(os.path.join(DATA, 'collapse-sqg-n60-no-rotation.json')))
        z = np.array([complex(float(p[0]), float(p[1])) for p in d['z']])
        G = np.array([float(x) for x in d['G']])
        a = 1.0
    N = len(z)
    z, G, b = normalize(z, G, a)
    z, G, ir, ig = gauge(z, G)
    z, G = rescale_G1(z, G, a, ig)
    u = pack(z, G, a, 0.0)
    S = System(u, N, free_mask(N, ir, ig, False, False))
    v, nf, ok = newton_project(S, u[S.idx], tol=1e-14)
    z, G, a, b = unpack(S.full(v), N)
    return z, G, a, nf


def invariants(z, G, a):
    N = len(z)
    d = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(d, 1.0)
    iu = np.triu_indices(N, 1)
    H = np.sum((G[:, None]*G[None, :]*d**(-a))[iu])
    Hs = np.sum(np.abs(G[:, None]*G[None, :]*d**(-a))[iu])
    return H, Hs, np.sum(G*z), np.sum(G*np.abs(z)**2), np.sum(np.abs(G)*np.abs(z)**2)


def size(z, G, zc):
    return np.sqrt(np.sum(np.abs(G)*np.abs(z - zc)**2)/np.sum(np.abs(G)))


def integrate(z0, G, a, pert, seed, tfrac, rtol=1e-13, npts=4000):
    N = len(z0)
    rng = np.random.default_rng(seed)
    R0 = size(z0, G, 0)
    dz = rng.normal(size=N) + 1j*rng.normal(size=N)
    dz *= pert*R0/np.sqrt(np.mean(np.abs(dz)**2))
    zi = z0 + dz
    tc = TWO_PI/(a + 2)
    H0, Hs, M0, I0, Is = invariants(zi, G, a)
    zc = M0/np.sum(G)

    def rhs(t, y):
        V = vel(y[:N] + 1j*y[N:], G, a)
        return np.concatenate([V.real, V.imag])
    tev = np.linspace(0, tfrac*tc, npts)
    sol = solve_ivp(rhs, (0, tfrac*tc), np.concatenate([zi.real, zi.imag]), method='DOP853', rtol=rtol, atol=1e-14*R0, t_eval=tev)
    zs = sol.y[:N].T + 1j*sol.y[N:].T
    s0 = size(zi, G, zc)
    sizes = np.array([size(zz, G, zc) for zz in zs])/s0
    w0 = zi - zc
    angs = np.unwrap(np.array([np.angle(np.sum(np.abs(G)*np.conj(w0)*(zz - zc))) for zz in zs]))
    dev = []
    for zz in zs:
        w = zz - zc
        c = np.sum(np.abs(G)*np.conj(w0)*w)/np.sum(np.abs(G)*np.abs(w0)**2)
        dev.append(np.sqrt(np.sum(np.abs(G)*np.abs(w - c*w0)**2)/np.sum(np.abs(G)*np.abs(w)**2)))
    dev = np.array(dev)
    inv = [invariants(zz, G, a) for zz in zs]
    ib = int(np.argmax(dev > 1e-2)) if np.any(dev > 1e-2) else len(dev) - 1
    imin = int(np.argmin(sizes))
    dmin = min(np.min(np.abs(zz[:, None] - zz[None, :])[np.triu_indices(N, 1)]) for zz in zs)
    exact = np.clip(1 - sol.t/tc, 0, None)**(1/(a + 2))
    return dict(pert=pert, seed=seed, status=int(sol.status), t_end=float(sol.t[-1]/tc), departed=bool(np.any(dev > 1e-2)),
                t_departure=float(sol.t[ib]/tc), size_at_departure=float(sizes[ib]), exact_size_at_departure=float(exact[ib]),
                angle_at_departure=float(angs[ib]), min_size=float(sizes[imin]), t_min_size=float(sol.t[imin]/tc),
                final_size=float(sizes[-1]), max_abs_angle=float(np.max(np.abs(angs))), min_pair_over_R0=float(dmin/R0),
                Hdrift=float(max(abs(x[0] - H0) for x in inv)/Hs), Mdrift=float(max(abs(x[2] - M0) for x in inv)/(np.sum(np.abs(G))*R0)),
                Idrift=float(max(abs(x[3] - I0) for x in inv)/Is))


say('Linear stability and perturbed integration of the collapses without rotation (NUMERICAL, binary64)%s' % ('  [--quick]' if QUICK else ''))
EXPECT = {'alpha2-n11': (16, 8, 258.15, 8.43, 1.5), 'sqg-n60': (114, 57, 4273.65, 5.04, 1.2)}
for case in ('alpha2-n11', 'sqg-n60'):
    z, G, a, nf = load(case)
    N = len(z)
    nshape, nunst, kmax, kmin, tfrac = EXPECT[case]
    say('\n--- %s: N = %d, alpha = %g; Newton polish residual %.1e  (%.0f s)' % (case, N, a, nf, time.time() - T0))
    dg = diagnostics(z, G, a)
    say('   closest pair / size %.4g, circulation span %.4g, relative angular impulse %.1e' % (dg['minpair_rel'], dg['Gspan'], dg['I']))
    J = jac_full(pack(z, G, a, 0.0), N)[:, :2*N]
    ev = list(np.linalg.eigvals(J))
    sym = [0.0, a + 2, 1.0, 1.0, 2.0, -a]
    found = []
    for s in sym:
        i = int(np.argmin([abs(e - s) for e in ev]))
        found.append(ev.pop(i))
    check('%s: the six symmetry exponents 0, 2 beta = %g, 1, 1, 2, -alpha = %g are present (largest error %.1e)' %
          (case, a + 2, -a, max(abs(f - s) for f, s in zip(found, sym))), max(abs(f - s) for f, s in zip(found, sym)) < 1e-9)
    evl = np.array(ev)
    pr = np.sort(evl.real)
    pairing = np.max(np.abs(pr + pr[::-1] - 2))
    check('%s: the exponents pair as k + k\' = 2 (Hamiltonian structure), largest defect %.1e' % (case, pairing), pairing < 1e-9)
    unst = sorted(evl[evl.real > 1e-9], key=lambda e: -e.real)
    ncomplex = int(np.sum(np.abs(evl.imag) > 1e-9))
    check('%s: %d shape modes, %d unstable (Re k > 0), all real (%d complex); largest k = %.2f, least positive k = %.2f' %
          (case, len(evl), len(unst), ncomplex, unst[0].real, unst[-1].real),
          len(evl) == nshape and len(unst) == nunst and ncomplex == 0 and abs(unst[0].real - kmax) < 0.01 and abs(unst[-1].real - kmin) < 0.01)
    say('   unstable k: ' + ', '.join('%.2f' % e.real for e in unst[:8]) + (' ...' if len(unst) > 8 else ''))
    r_pred = 1e4**(-1/unst[0].real)
    say('   growth at the largest exponent alone from 1e-6 to 1e-2 predicts departure at size r = 1e4^(-1/k_max) = %.4f' % r_pred)
    runs = []
    for pert, seed in [(0.0, 0), (1e-6, 1), (1e-6, 2), (1e-6, 3)]:
        tf = tfrac if not (QUICK and case == 'sqg-n60') else 0.08
        t1 = time.time()
        r = integrate(z, G, a, pert, seed, tf)
        runs.append(r)
        say('   integration pert %.0e seed %d to %.2f t_c: departs at t/t_c %.4f, size %.4f (exact %.4f), rotation %.1e rad; least size %.4f at t/t_c %.3f; '
            'final size %.4f; largest |rotation| %.3f; closest pair / R0 %.3g; drift H %.0e, M %.0e, I %.0e  (%.0f s)' %
            (pert, seed, tf, r['t_departure'], r['size_at_departure'], r['exact_size_at_departure'], r['angle_at_departure'], r['min_size'],
             r['t_min_size'], r['final_size'], r['max_abs_angle'], r['min_pair_over_R0'], r['Hdrift'], r['Mdrift'], r['Idrift'], time.time() - t1))
    pert_runs = [r for r in runs if r['pert'] > 0]
    check('%s: every integration completes, conserves H, the impulse and the angular impulse (largest relative drifts %.0e, %.0e, %.0e)' %
          (case, max(r['Hdrift'] for r in runs), max(r['Mdrift'] for r in runs), max(r['Idrift'] for r in runs)),
          all(r['status'] == 0 for r in runs) and max(r['Hdrift'] for r in runs) < 1e-8 and max(r['Idrift'] for r in runs) < 1e-10)
    check('%s: every run departs from the self-similar collapse; the perturbed ones at sizes %s' %
          (case, ', '.join('%.4f' % r['size_at_departure'] for r in pert_runs)),
          all(r['departed'] for r in runs) and all(r['size_at_departure'] > r_pred - 0.02 for r in pert_runs))
    check('%s: until departure the motion follows the collapse without turning: |rotation| < 1e-4 rad in every run (largest %.1e)' %
          (case, max(abs(r['angle_at_departure']) for r in runs)), max(abs(r['angle_at_departure']) for r in runs) < 1e-4)
    if not (QUICK and case == 'sqg-n60'):
        check('%s: after departure no perturbed run gets below %.3f of its initial size, and no close pair forms (closest pair / R0 >= %.3g)' %
              (case, min(r['min_size'] for r in pert_runs), min(r['min_pair_over_R0'] for r in pert_runs)),
              min(r['min_size'] for r in pert_runs) > 0.8 and min(r['min_pair_over_R0'] for r in pert_runs) > 1e-3)
    out = dict(case=case, N=N, alpha=a, exponents=[[e.real, e.imag] for e in evl], unstable=[e.real for e in unst],
               z=[[p.real, p.imag] for p in z], G=list(map(float, G)), runs=runs, quick=QUICK)
    json.dump(out, open(os.path.join(DATA, 'stability-%s%s.json' % (case, '-quick' if QUICK else '')), 'w'), indent=1)

say('\n%d checks, %d failed  (%.0f s)' % (NCHK[0], len(FAILED), time.time() - T0))
with open(os.path.join(DATA, 'verify-stability%s.txt' % ('-quick' if QUICK else '')), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
