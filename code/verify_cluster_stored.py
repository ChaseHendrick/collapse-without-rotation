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
"""An independent recheck of the stored cluster collapses, data/cluster-collapses.json (written by
verify_cluster_collapses.py, 40 digits), with code that shares nothing with the program that computed them.

For every stored collapse, at 60 digits: the velocities of all vortices from the Euler law; kappa by least squares
over all vortices about the centre of vorticity (not from one vortex); the residual max_j |v_j - kappa (z_j - z_c)|
relative to max |v|; the two invariants that every self-similar Euler collapse makes vanish, sum_{i<j} Gamma_i Gamma_j
and the angular impulse sum Gamma_j |z_j - z_c|^2; P = |Im kappa|/(-2 Re kappa) against the stored value; the
largest c for which the configuration lies in the class K(n, c) of Theorem 1 with the stored partition into
clusters and Z_C the centroid; and, for one cluster, the net-circulation ratio nuhat = sum_C Gamma_k/((1/2) sum_C
Gamma_k^2) against the exact consequence 1 - gamma^2 nu^2/S2 of sum_{i<j} Gamma_i Gamma_j = 0, and nuhat - 1 against
-gamma^2 S2/4.

Needs mpmath. Run: python3 code/verify_cluster_stored.py (after verify_cluster_collapses.py). Prints every check and
exits with status 1 if any fails; output in data/verify-cluster-stored.txt. A few seconds.
"""
import os
import sys
import json
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
mp.mp.dps = 60
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


rec = json.load(open(os.path.join(DATA, 'cluster-collapses.json')))['collapses']
say('Independent recheck of %d stored collapses (60 digits; stored values have 40)' % len(rec))
worst_res, worst_P, worst_inv, least_c = mp.mpf(0), mp.mpf(0), mp.mpf(0), mp.mpf(10)
for r in rec:
    G = [mp.mpf(v) for v in r['circulations']]
    Z = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in r['positions']]
    N = len(Z)
    gam = mp.mpf(r['gamma'])
    S = mp.fsum(G)
    zc = mp.fsum(g*z for g, z in zip(G, Z))/S
    v = [mp.conj(mp.fsum(G[k]/(Z[j] - Z[k]) for k in range(N) if k != j)/(2j*mp.pi)) for j in range(N)]
    kap = mp.fsum(v[j]*mp.conj(Z[j] - zc) for j in range(N))/mp.fsum(abs(Z[j] - zc)**2 for j in range(N))
    res = max(abs(v[j] - kap*(Z[j] - zc)) for j in range(N))/max(abs(x) for x in v)
    P = abs(kap.imag)/(-2*kap.real)
    s2 = mp.fsum(G[i]*G[j] for i in range(N) for j in range(i))/mp.fsum(abs(G[i]*G[j]) for i in range(N) for j in range(i))
    L = mp.fsum(G[i]*abs(Z[i] - zc)**2 for i in range(N))/mp.fsum(abs(G[i])*abs(Z[i] - zc)**2 for i in range(N))
    # class membership: the largest c with (Z_C = centroid of cluster C, g = Gamma/gamma)
    cands = []
    cents = []
    for mem in r['clusters']:
        idx = [k + 1 for k in mem]
        Zc = mp.fsum(Z[k] for k in idx)/len(idx)
        cents.append(Zc)
        for k in idx:
            gk = abs(G[k])/gam
            cands += [gk, 1/gk]
            dk = abs(Z[k] - Zc)
            if dk > 0:
                cands.append(gam/dk)
        for a in idx:
            for b in idx:
                if a < b:
                    cands.append(abs(Z[a] - Z[b])/gam)
        cands += [abs(Zc), 1/abs(Zc)]
    for i in range(len(cents)):
        for j in range(i):
            cands.append(abs(cents[i] - cents[j]))
    cmax = min(min(cands), mp.mpf(1))
    worst_res = max(worst_res, res)
    worst_P = max(worst_P, abs(P - mp.mpf(r['P'])))
    worst_inv = max(worst_inv, abs(s2), abs(L))
    least_c = min(least_c, cmax)
    line = '%-55s gamma %-5s P - sqrt3/2 = %s  residual %s  sum G_iG_j %s  impulse %s  in K(n, c) for c <= %s' % (
        r['case'][:55], r['gamma'], mp.nstr(P - mp.sqrt(3)/2, 8), mp.nstr(res, 3), mp.nstr(s2, 3), mp.nstr(L, 3), mp.nstr(cmax, 4))
    if len(r['clusters']) == 1:
        idx = [k + 1 for k in r['clusters'][0]]
        g = [G[k]/gam for k in idx]
        nu = mp.fsum(g)/gam
        S2 = mp.fsum(x**2 for x in g)
        nuhat = 2*nu/S2
        exact = 1 - gam**2*nu**2/S2
        ratio = (nuhat - 1)/(-gam**2*S2/4)
        check(line + '; nuhat - (1 - gamma^2 nu^2/S2) = %s, (nuhat - 1)/(-gamma^2 S2/4) = %s' % (mp.nstr(nuhat - exact, 3), mp.nstr(ratio, 10)),
              abs(nuhat - exact) < mp.mpf(10)**-30 and abs(ratio - 1) < 3*gam**2*S2)
    else:
        check(line, res < mp.mpf(10)**-35)
check('every stored collapse is self-similar: least-squares kappa over all vortices, largest residual %s < 1e-35' % mp.nstr(worst_res, 3),
      worst_res < mp.mpf(10)**-35)
check('P recomputed independently agrees with the stored value to %s' % mp.nstr(worst_P, 3), worst_P < mp.mpf(10)**-35)
check('sum_{i<j} Gamma_i Gamma_j and the angular impulse vanish (relative, largest %s)' % mp.nstr(worst_inv, 3), worst_inv < mp.mpf(10)**-35)
check('every stored collapse lies in K(n, c) for c = %s (the least over all), with the stored clusters' % mp.nstr(least_c, 4), least_c >= mp.mpf('0.2'))
bycase = {}
for r in rec:
    bycase.setdefault(r['case'], {})[r['gamma']] = mp.mpf(r['P'])
below = [c for c, d in bycase.items() if d['1e-4'] < mp.sqrt(3)/2]
check('P < sqrt3/2 at gamma = 1e-4 exactly for the arrangements with e_C > 0 for every cluster: %s' % '; '.join(below),
      set(below) == {c for c in bycase if 'pair' not in c})

say('\n%d checks, %d failed' % (NCHK[0], len(FAILED)))
with open(os.path.join(DATA, 'verify-cluster-stored.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
