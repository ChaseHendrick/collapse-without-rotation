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
"""The nine-parameter family of collapses without rotation of eleven vortices at alpha = 2 (Sections 5.1 and 5.2 of
paper/collapse-without-rotation.tex). NUMERICAL except item 3, which illustrates Proposition 1 (proved in the paper).

  1. At the collapse of the companion paper's Theorem 5(a) (data/collapse-alpha2-n11-no-rotation.json, refined by
     Newton's method in binary64), the 22 equations have full rank in 31 unknowns: the family has dimension 9.
  2. Continuation along both signs of each of the nine tangent directions (up to 300 steps): no collision, no
     vanishing circulation, no loss of rank; the sign pattern of the circulations never changes.
  3. Proposition 1 (no mirror symmetry): for random configurations and alpha = 0, 0.5, 1, 2 the velocity of the
     mirror image is minus the conjugate velocity, so the mirror image of a collapse is an expansion; the mirror image
     of the alpha = 2 collapse satisfies V = kappa' z with kappa' = -conj(kappa); and neither the collapse nor any end
     point of item 2 is mirror symmetric (their mirror defect is bounded away from 0), while every total circulation
     is nonzero.
  4. No C5 member: a C_n-symmetric configuration of two n-gons and a centre (N = 2n + 1) reduces to one scalar
     equation F(rho, theta) = 0 once the ring circulation -1/rho^2 (equal radial rates) and the central circulation
     (linear) are eliminated; the reduction is checked against the full Biot-Savart velocities, and on a 200 x 200
     grid (0.05 <= rho <= 20, 0 < theta < 2 pi/n) F has one sign for n = 2, 3, 5 at alpha = 1 and 2.
  5. The most nearly equal weak circulations found (a failed attempt at a simplest member): their spread.

Needs numpy and scipy. Run: python3 code/verify_family_structure.py. Prints every check and exits with status 1 if
any fails; output in data/verify-family-structure.txt. About a minute.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from collapse_core import vel, pack, unpack, System, normalize, gauge, rescale_G1, free_mask, newton_project, null_basis, diagnostics, TWO_PI  # noqa: E402

warnings.filterwarnings('ignore')
DATA = os.path.join(os.path.dirname(HERE), 'data')
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


def mirror_defect(z, G, sign=1):
    """Least over reflection lines through 0 of the matching defect between the mirror image (circulations times
    sign) and the configuration."""
    best = 9.0
    for phi in np.linspace(0, np.pi, 721):
        zz = np.exp(2j*phi)*np.conj(z)
        C = np.abs(zz[:, None] - z[None, :])/np.max(abs(z)) + np.abs(sign*G[:, None] - G[None, :])/np.max(abs(G))
        r, c = linear_sum_assignment(C)
        best = min(best, C[r, c].max())
    return best


say('The collapses without rotation of eleven vortices at alpha = 2 (NUMERICAL, binary64)')
d = json.load(open(os.path.join(DATA, 'collapse-alpha2-n11-no-rotation.json')))
m = d['minima'][0]['config']
z0, G0 = np.array([complex(*p) for p in m['z']]), np.array(m['G'], float)
a = 2.0
N = len(z0)
z0, G0, b = normalize(z0, G0, a)
z0, G0, ir, ig = gauge(z0, G0)
z0, G0 = rescale_G1(z0, G0, a, ig)
u0 = pack(z0, G0, a, 0.0)
S = System(u0, N, free_mask(N, ir, ig, False, False))
v0, nf, ok = newton_project(S, u0[S.idx], tol=1e-14)
z0, G0, _, _ = unpack(S.full(v0), N)

say('\n1. The dimension of the family')
J = S.J(v0)
Zb, s = null_basis(J)
rank = int(np.sum(s > s[0]*1e-12))
check('at the collapse (Newton residual %.1e): %d equations, %d unknowns, rank %d, tangent dimension %d, sigma_min/sigma_max %.2e' %
      (nf, J.shape[0], J.shape[1], rank, Zb.shape[1], s[-1]/s[0]), J.shape == (22, 31) and rank == 22 and Zb.shape[1] == 9)
strong = np.argsort(-np.abs(G0))[:2]
signs0 = ''.join('+' if g > 0 else '-' for g in G0)
say('   circulations (strongest = -1): %s; signs %s' % (np.round(G0, 4), signs0))

say('\n2. Continuation along the nine tangent directions  (%.0f s)' % (time.time() - T0))
ends = []
for i in range(Zb.shape[1]):
    for sgn in (1, -1):
        v = v0.copy()
        dvec = sgn*Zb[:, i]
        h = 0.02*np.linalg.norm(v0)
        reason, L = 'max steps', 0.0
        spans, pairs = [], []
        for step in range(300):
            vn, nfn, okn = newton_project(S, v + h*dvec, maxit=20)
            if not okn:
                h /= 2
                if h < 1e-6*np.linalg.norm(v0):
                    reason = 'projection failed'
                    break
                continue
            Zn, sn = null_basis(S.J(vn))
            if Zn.shape[1] != Zb.shape[1]:
                reason = 'rank change'
                break
            dn = Zn @ (Zn.T @ dvec)
            if np.linalg.norm(dn) < 1e-8:
                reason = 'direction lost'
                break
            dvec = dn/np.linalg.norm(dn)
            L += np.linalg.norm(vn - v)
            v = vn
            zz, GG, _, _ = unpack(S.full(v), N)
            dg = diagnostics(zz, GG, a)
            spans.append(dg['Gspan'])
            pairs.append(dg['minpair_rel'])
            if dg['minpair_rel'] < 2e-3:
                reason = 'collision'
                break
            if np.min(np.abs(GG))/np.max(np.abs(GG)) < 1e-3:
                reason = 'a circulation -> 0'
                break
            if dg['Rmin'] < 1e-3*dg['Rmax']:
                reason = 'a vortex at the collision point'
                break
        zz, GG, _, _ = unpack(S.full(v), N)
        ends.append(dict(dir=i, sgn=sgn, L=L/np.linalg.norm(v0), reason=reason, signs=''.join('+' if g > 0 else '-' for g in GG),
                         span=(min(spans), max(spans)), pair=min(pairs), z=zz, G=GG, res=float(np.linalg.norm(S.F(v)))))
        say('   direction %d%+d: arclength %.2f, end: %s, closest pair >= %.3f, circulation span %.1f..%.1f, signs %s' %
            (i, sgn, ends[-1]['L'], reason, min(pairs), min(spans), max(spans), ends[-1]['signs']))
check('all 18 continuations run to the step limit (arclength %.1f to %.1f relative to the configuration) without collision, vanishing '
      'circulation or loss of rank' % (min(e['L'] for e in ends), max(e['L'] for e in ends)), all(e['reason'] == 'max steps' for e in ends))
check('the sign pattern of the circulations (one opposite-signed strong vortex, ten of the other sign) never changes',
      all(e['signs'] == signs0 for e in ends) and signs0.count('+') in (1, 10))
lo = min(e['span'][0] for e in ends)
hi = max(e['span'][1] for e in ends)
check('along the continuations the circulation span varies from %.0f to %.0f and the end points solve the equations (largest residual %.1e)' %
      (lo, hi, max(e['res'] for e in ends)), hi > 2*lo and max(e['res'] for e in ends) < 1e-10)

say('\n3. Proposition 1: mirror images  (%.0f s)' % (time.time() - T0))
rng = np.random.default_rng(11)
worst = 0.0
for al in (0.0, 0.5, 1.0, 2.0):
    for t in range(50):
        n = int(rng.integers(3, 12))
        z = rng.normal(size=n) + 1j*rng.normal(size=n)
        G = rng.normal(size=n)
        worst = max(worst, np.max(np.abs(vel(np.conj(z), G, al) + np.conj(vel(z, G, al))))/np.max(np.abs(vel(z, G, al))))
check('V(mirror image, same circulations) = -conj(V) for 200 random configurations, alpha = 0, 0.5, 1, 2 (largest relative defect %.1e): '
      'the mirror image of a solution is a solution run backwards' % worst, worst < 1e-12)
Vm = vel(np.conj(z0), G0, a)
kap = np.sum(Vm*np.conj(np.conj(z0)))/np.sum(np.abs(z0)**2)
resm = np.max(np.abs(Vm - kap*np.conj(z0)))/np.max(np.abs(Vm))
check('the mirror image of the alpha = 2 collapse (2 pi kappa = -1) moves with 2 pi kappa\' = %.12f%+.1ei = -conj(2 pi kappa): an expansion (residual %.1e)' %
      (TWO_PI*kap.real, TWO_PI*kap.imag, resm), abs(TWO_PI*kap - 1) < 1e-10 and resm < 1e-10)
md = [mirror_defect(z0, G0)] + [mirror_defect(e['z'], e['G']) for e in ends[::3]]
check('neither the collapse nor the end points of item 2 are mirror symmetric: least mirror defect %.3f (> 0), and every total circulation is '
      'nonzero (least |sum G|/sum|G| %.3f)' % (min(md), min(abs(np.sum(e['G']))/np.sum(np.abs(e['G'])) for e in ends)),
      min(md) > 0.01 and min(abs(np.sum(e['G']))/np.sum(np.abs(e['G'])) for e in ends) > 0.01)

say('\n4. No C_n-symmetric member: two n-gons and a centre  (%.0f s)' % (time.time() - T0))


def conf(n, rho, th, g0):
    k = np.arange(n)
    e = np.exp(2j*np.pi*k/n)
    return np.concatenate([e, rho*np.exp(1j*th)*e, [0]]), np.concatenate([np.ones(n), -np.ones(n)/rho**2, [g0]])


def tang(n, rho, th, al, g0):
    z, G = conf(n, rho, th, g0)
    V = vel(z, G, al)
    return ((np.conj(z[0])*V[0]).imag/abs(z[0]), (np.conj(z[n])*V[n]).imag/abs(z[n]),
            (np.conj(z[0])*V[0]).real/abs(z[0])**2, (np.conj(z[n])*V[n]).real/abs(z[n])**2)


def Ffun(n, rho, th, al):
    t10, t20, r1, r2 = tang(n, rho, th, al, 0.0)
    t11, t21, _, _ = tang(n, rho, th, al, 1.0)
    t11 -= t10
    t21 -= t20
    return t10*t21 - t20*t11, t11, t10, abs(t10*t21) + abs(t20*t11)


worst_red = 0.0
for t in range(40):
    n = int(rng.choice([2, 3, 5]))
    al = float(rng.choice([1.0, 2.0]))
    rho = float(np.exp(rng.uniform(np.log(0.1), np.log(10))))
    th = float(rng.uniform(0.01, 2*np.pi/n - 0.01))
    F, t11, t10, sc = Ffun(n, rho, th, al)
    g0 = -t10/t11
    z, G = conf(n, rho, th, g0)
    V = vel(z, G, al)
    rate = (np.conj(z)*V).real/np.abs(z)**2
    tan_ = (np.conj(z)*V).imag/np.abs(z)**2
    k = np.arange(2*n)
    # with this g0: ring 1 has no tangential velocity, ring 2 has tangential velocity -F/t11 (times 1/rho), equal radial rates
    d1 = np.max(np.abs(tan_[:n]))
    d2 = np.max(np.abs(tan_[n:2*n]*rho - (-F/t11)))
    d3 = np.max(np.abs(rate[:2*n] - rate[0]))
    scale = np.max(np.abs(V[:2*n])/np.abs(z[:2*n]))
    worst_red = max(worst_red, d1/scale, d2/(scale*rho), d3/scale)
check('the reduction against the full Biot-Savart velocities at 40 random points: with the central circulation eliminated, ring 1 turns '
      'not at all, ring 2 turns at the rate -F/t11, and all radial rates agree (largest defect relative to the velocity scale %.1e)' % worst_red, worst_red < 1e-12)
for al in (1.0, 2.0):
    for n in (2, 3, 5):
        rhos = np.exp(np.linspace(np.log(0.05), np.log(20), 200))
        ths = np.linspace(1e-3, 2*np.pi/n - 1e-3, 200)
        FF = np.zeros((200, 200))
        rel = np.zeros((200, 200))
        for i, rho in enumerate(rhos):
            for j, th in enumerate(ths):
                F, _, _, sc = Ffun(n, rho, th, al)
                FF[i, j] = F
                rel[i, j] = F/sc if sc > 0 else 0
        changes = 0
        for i in range(199):
            for j in range(199):
                blk = FF[i:i + 2, j:j + 2]
                if blk.min() < 0 < blk.max():
                    changes += 1
        check('alpha = %g, n = %d (N = %d): F has one sign on the grid (%d sign changes; F/scale between %.2e and %.2e)' %
              (al, n, 2*n + 1, changes, rel.min(), rel.max()), changes == 0 and (FF.max() < 0 or FF.min() > 0))

say('\n5. The most nearly equal weak circulations found  (%.0f s)' % (time.time() - T0))
fam = json.load(open(os.path.join(DATA, 'searches', 'family-alpha2-n11.json')))
ew = fam['equal_weak']
zE, GE = np.array([complex(*p) for p in ew['z']]), np.array(ew['G'])
V = vel(zE, GE, a)
res = np.max(np.abs(TWO_PI*V + zE))/np.max(np.abs(zE))
weak = np.argsort(np.abs(GE))[:9]
ratio = np.abs(GE[weak]).max()/np.abs(GE[weak]).min()
check('the member with the most nearly equal nine weak circulations found is a collapse without rotation (residual %.1e); its weak '
      'circulations still differ by a factor %.2f, so no equal-circulation member was found' % (res, ratio), res < 1e-10 and ratio > 1.2)

say('\n%d checks, %d failed  (%.0f s)' % (NCHK[0], len(FAILED), time.time() - T0))
with open(os.path.join(DATA, 'verify-family-structure.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
