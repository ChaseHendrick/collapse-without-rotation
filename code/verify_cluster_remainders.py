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
"""The remainders in the proof of Theorem 1 of paper/collapse-without-rotation.tex, on random configurations of the
class K(n, c) (any Lambda; these are identities with remainders, not properties of solutions), and the explicit
constants of Step 1 and of part (d).

With F_k = sum_{l != k} Gamma_l/(z_k - z_l) - Lambda conj(z_k - z_c) (zero at a collapse), T = Lambda conj(Z - z_c) - 1/Z - E(Z):
  (A') sum_{k in C} g_k F_k + G T + gamma (Lambda conj D + D/Z^2)  = O(gamma^2),
  (B)  sum_{k in C} g_k zeta_k F_k - [(G^2 - S2)/2 - D T]         = O(gamma),
  (W)  w_k = T + F_k + O(gamma), so that at a collapse (every F_k = 0) the internal field of each cluster is a
       translating relative equilibrium up to O(gamma): max_{k in C} |w_k - F_k - T| = O(gamma).
Each remainder divided by its order must stay bounded as gamma -> 0 (1e-2 .. 1e-7); a negative control that drops
the term gamma D/Z^2 from (A') must grow like 1/gamma.
Step 1 and part (d), on random configurations at the largest admissible gamma: |z_c| <= 4 n gamma/c^2 for
gamma <= c^3/(16 n); for a singleton k and gamma <= c^3/(20 n), |e1| = |z_c/z_k| <= 1.2 n gamma/c^3 and
|e2| = |z_k E_k| <= n gamma/c^3, for n up to 40.

Needs mpmath (code/requirements.txt). Run: python3 code/verify_cluster_remainders.py. Prints every check and exits
with status 1 if any fails; output in data/verify-cluster-remainders.txt. A few seconds.
"""
import os
import sys
import random
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
mp.mp.dps = 40
random.seed(7)
OUT = []
FAILED = []
NCHK = [0]


def say(s=''):
    print(s)
    OUT.append(s)


def check(name, ok, detail=''):
    NCHK[0] += 1
    say(('OK    ' if ok else 'FAIL  ') + name + (('   [' + detail + ']') if detail else ''))
    if not ok:
        FAILED.append(name)


say('Remainders of the weighted sums (A\'), (B) and of the TRE defect (W) on random configurations of K(n, c), c = 0.3')
c = mp.mpf('0.3')
worstA, worstB, worstW, worstNeg = {}, {}, {}, {}
for trial in range(150):
    M = random.randint(1, 3)
    sizes = [random.randint(1, 4) for _ in range(M)]
    ang = [2*mp.pi*C/M + mp.mpf(random.uniform(-0.3, 0.3)) for C in range(M)]
    Zs = [mp.mpf(random.uniform(0.5, 2.5))*mp.exp(1j*a) for a in ang]
    shapes = []
    for s in sizes:
        while True:
            pts = [mp.mpc(random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)) for _ in range(s)]
            if all(abs(pts[i] - pts[j]) > c for i in range(s) for j in range(i)):
                break
        gs = [mp.mpf(random.choice([-1, 1])*random.uniform(0.4, 2.5)) for _ in range(s)]
        shapes.append((pts, gs))
    Lam = mp.mpc(random.uniform(-3, 3), random.uniform(-3, 3))
    for gexp in range(2, 8):
        gam = mp.mpf(10)**(-gexp)
        zs, Gs, owner = [mp.mpc(0)], [mp.mpf(1)], [-1]
        for C, (pts, gs) in enumerate(shapes):
            for p_, g_ in zip(pts, gs):
                zs.append(Zs[C] + gam*p_)
                Gs.append(gam*g_)
                owner.append(C)
        zc = mp.fsum(a*b for a, b in zip(Gs, zs))/mp.fsum(Gs)
        F = [mp.fsum(Gs[l]/(zs[k] - zs[l]) for l in range(len(zs)) if l != k) - Lam*mp.conj(zs[k] - zc) for k in range(len(zs))]
        for C in range(M):
            mem = [k for k in range(len(zs)) if owner[k] == C]
            Z = Zs[C]
            g = [Gs[k]/gam for k in mem]
            zeta = [(zs[k] - Z)/gam for k in mem]
            G = mp.fsum(g)
            D = mp.fsum(a*b for a, b in zip(g, zeta))
            S2 = mp.fsum(a**2 for a in g)
            E = mp.fsum(Gs[l]/(Z - zs[l]) for l in range(1, len(zs)) if owner[l] != C)
            Tt = Lam*mp.conj(Z - zc) - 1/Z - E
            RA = mp.fsum(g[i]*F[k] for i, k in enumerate(mem)) + G*Tt + gam*(Lam*mp.conj(D) + D/Z**2)
            RB = mp.fsum(g[i]*zeta[i]*F[k] for i, k in enumerate(mem)) - ((G**2 - S2)/2 - D*Tt)
            Rneg = RA - gam*D/Z**2
            # (W): w_k = T + F_k + O(gamma) for every k in C, so the spread of w_k - F_k over C is O(gamma)
            w = [mp.fsum(g[j]/(zeta[i] - zeta[j]) for j in range(len(mem)) if j != i) for i in range(len(mem))]
            vals = [w[i] - F[k] for i, k in enumerate(mem)]
            RW = max(abs(Tt - v) for v in vals)
            worstA[gexp] = max(worstA.get(gexp, 0), abs(RA)/gam**2)
            worstB[gexp] = max(worstB.get(gexp, 0), abs(RB)/gam)
            worstW[gexp] = max(worstW.get(gexp, 0), RW/gam)
            worstNeg[gexp] = max(worstNeg.get(gexp, 0), abs(Rneg)/gam**2)
for gexp in sorted(worstA):
    say('gamma=1e-%d: max|R_A|/gamma^2 = %s   max|R_B|/gamma = %s   max|w_k - F_k - T|/gamma = %s   control max|R|/gamma^2 = %s' %
        (gexp, mp.nstr(worstA[gexp], 5), mp.nstr(worstB[gexp], 5), mp.nstr(worstW[gexp], 5), mp.nstr(worstNeg[gexp], 5)))
check("(A') remainder / gamma^2 does not grow as gamma -> 0 (150 configurations, 1-3 clusters of 1-4 vortices, mixed signs)",
      worstA[7] < 2*worstA[3])
check('(B) remainder / gamma does not grow as gamma -> 0', worstB[7] < 2*worstB[3])
check('(W) w_k = T + F_k + O(gamma), so at a collapse (F = 0) every cluster is a translating relative equilibrium up to O(gamma)',
      worstW[7] < 2*worstW[3])
check('negative control (drop gamma D/Z^2 from (A\')): remainder / gamma^2 grows like 1/gamma', worstNeg[7] > 1e3*worstNeg[3])

say('\nStep 1 and part (d): the explicit constants at the largest admissible gamma')
worst_zc, worst_e1, worst_e2 = 0, 0, 0
placed = 0
for trial in range(400):
    n = random.choice([1, 2, 3, 5, 8, 13, 20, 40])
    cc = mp.mpf(random.choice(['0.1', '0.3', '0.6', '1'] if n <= 3 else ['0.1', '0.3', '0.6'] if n <= 5 else ['0.1', '0.3']))
    gam = cc**3/(20*n)
    # the singleton: vortex 1 alone in its cluster, the others in clusters whose centres are >= c apart
    M = n
    cen = []
    tries = 0
    while len(cen) < M and tries < 3000:
        tries += 1
        r = mp.mpf(random.uniform(float(cc), float(1/cc)))
        Zc = r*mp.expj(random.uniform(0, 2*3.141592653589793))
        if all(abs(Zc - X) >= cc for X in cen):
            cen.append(Zc)
    if len(cen) < M:
        continue
    placed += 1
    zs, Gs = [mp.mpc(0)], [mp.mpf(1)]
    for X in cen:
        off = mp.mpf(random.uniform(0, 1))*gam/cc*mp.expj(random.uniform(0, 6.283))
        zs.append(X + off)
        Gs.append(gam*random.choice([-1, 1])*mp.mpf(random.uniform(float(cc), float(1/cc))))
    zc = mp.fsum(a*b for a, b in zip(Gs, zs))/mp.fsum(Gs)
    k = 1
    Ek = mp.fsum(Gs[l]/(zs[k] - zs[l]) for l in range(2, len(zs)))
    e1 = abs(zc/zs[k])
    e2 = abs(zs[k]*Ek)
    worst_zc = max(worst_zc, abs(zc)/(4*n*gam/cc**2))
    worst_e1 = max(worst_e1, e1/(1.2*n*gam/cc**3))
    worst_e2 = max(worst_e2, e2/(n*gam/cc**3))
check('%d random configurations with n = 1..40 weak vortices placed' % placed, placed >= 300)
check('|z_c| <= 4 n gamma/c^2 (largest ratio %s)' % mp.nstr(worst_zc, 4), worst_zc <= 1)
check('singleton: |e1| <= 1.2 n gamma/c^3 (largest ratio %s), within 4 n gamma/c^3 used in part (d)' % mp.nstr(worst_e1, 4), worst_e1 <= 1)
check('singleton: |e2| <= n gamma/c^3 (largest ratio %s), for n up to 40' % mp.nstr(worst_e2, 4), worst_e2 <= 1)

say('\n%d checks, %d failed' % (NCHK[0], len(FAILED)))
with open(os.path.join(DATA, 'verify-cluster-remainders.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
