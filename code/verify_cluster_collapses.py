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
"""Exact self-similar collapses of a strong Euler vortex with weak tight clusters (Theorem 1 and the formal
expansion of Section 3.3 of paper/collapse-without-rotation.tex), computed by Newton's method at 50 digits and
accepted only after a direct Biot-Savart check of every vortex.

A cluster C carries circulations gamma g_k at z_k = Z_C + O(gamma). Its shape is a translating relative
equilibrium (TRE) of zero net circulation, found first; the family is then parametrized by y = Im q of the first
cluster, q_C = -2 D_C/(S2_C Z_C), with the cluster circulations (all but the last of each cluster), the directions
of the cluster centres and Z_1 = 1 fixed. P_min is the least P along this one-parameter family: it is minimized
over y only, with the cluster shapes and directions held fixed.

Checks, all as CONVERGENCE TESTS over gamma = 1e-2, 1e-3, 1e-4 (the error must shrink by at least a factor 5 per
decade of gamma, where the claimed order is O(gamma) or better):
  1. seven cluster arrangements (pair; two triples; two quadruples; pair + triple; three clusters): Biot-Savart
     residual < 1e-40 and angular impulse < 1e-40 at every point used; the error of the first-order formula
     P_min = sqrt3/2 - (2/sqrt3) min_C e_C decreases like gamma^2 (ratio 0.05..0.2 per decade of the error of the
     slope); for the arrangements in which a pair binds (e = 0), P_min > sqrt3/2; the leading-order structure of
     part (b) of Theorem 1 (nuhat - 1, Re q - 1/2, the TRE defect, P - (y^2 + 3/4)/(2y)) converges to 0, also at
     y = 0.7 and y = 1.05 away from the minimum (reached by continuation in y);
  2. a (+,-,-) triple (e < 0): P_min > sqrt3/2, with the predicted slope; the closed form 2 sqrt3 g1 g2 g3/S2 of the
     slope for equilateral triples;
  3. part (d), singletons: rings of Demina and Kudryashov with a strong centre at fixed relative rotation theta, whose
     weak vortices are at O(1) distances: exact collapses with P gamma -> tan(n theta/2)/n and P >= c^3/(40 n gamma);
  4. controls: the two-arm minimizers of the companion paper (P < sqrt3/2) are outside the regime of Theorem 1 (two
     co-dominant vortices, so gamma >= c in any class that contains them; like-signed weak vortices); the seven-vortex collapse of Demina and
     Kudryashov (P = 0.8053...) lies in K(6, c) with gamma = c for every c <= 1/3, so the threshold gamma_1 of
     Theorem 1(a) cannot be uniform as c -> 0; the certified four-vortex minimum of the companion paper is a strong
     vortex with a (+,+,-) triple whose net circulation is close to (1/2) sum Gamma_k^2.

The collapses are written to data/cluster-collapses.json (40 digits), for verify_cluster_stored.py (not in --quick mode).
Needs mpmath (code/requirements.txt). Run: python3 code/verify_cluster_collapses.py [--quick]. The full run takes
about two minutes; --quick drops the three-cluster arrangement and the second quadruple (about a minute).
Prints every check and exits with status 1 if any fails; output in data/verify-cluster-collapses.txt.
"""
import os
import sys
import json
import time
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
QUICK = '--quick' in sys.argv
mp.mp.dps = 50
FAILED = []
NCHK = [0]
OUT = []
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


def w_internal(zeta, g, k):
    return mp.fsum(g[l]/(zeta[k] - zeta[l]) for l in range(len(g)) if l != k)


def find_TRE(g, seed_pts):
    """Translating relative equilibrium of circulations g (sum g = 0): all w_k equal. Gauge zeta_0 = 0, zeta_1 = 1."""
    n = len(g)

    def F(*v):
        zeta = [mp.mpc(0), mp.mpc(1)] + [mp.mpc(v[2*i], v[2*i + 1]) for i in range(n - 2)]
        w0 = w_internal(zeta, g, 0)
        out = []
        for k in range(1, n - 1):
            d = w_internal(zeta, g, k) - w0
            out += [d.real, d.imag]
        return out
    x0 = []
    for s in seed_pts:
        x0 += [mp.mpf(s.real), mp.mpf(s.imag)]
    sol = mp.findroot(F, x0, tol=mp.mpf(10)**(-44), maxsteps=200)
    sol = sol if hasattr(sol, '__len__') else [sol]
    zeta = [mp.mpc(0), mp.mpc(1)] + [mp.mpc(sol[2*i], sol[2*i + 1]) for i in range(n - 2)]
    return zeta, [w_internal(zeta, g, k) for k in range(n)]


def e_coef(g, zeta):
    """e_C/gamma = (sum g h^2)(sum g^2)/|sum g zeta|^2 = J S2/(2|D|^2), h the coordinate perpendicular to D."""
    D = mp.fsum(a*b for a, b in zip(g, zeta))
    al = D/abs(D)
    h = [mp.im(b/al) for b in zeta]
    return mp.fsum(a*hh**2 for a, hh in zip(g, h))*mp.fsum(a**2 for a in g)/abs(D)**2


class Family:
    """Unknowns: all weak positions, Lambda = 2 pi i conj(kappa) and the last circulation of every cluster.
    Equations: sum_k Gamma_k/(z_j - z_k) = Lambda conj(z_j - z_c) at every weak vortex; Z_1 = 1; arg Z_C = phi_C (C >= 2);
    Im q_1 = y. The strong vortex has circulation 1 at the origin; its own equation follows from the others."""

    def __init__(self, gam, clusters, phis):
        self.gam = mp.mpf(gam)
        self.cl = clusters
        self.phis = [mp.mpf(p) for p in phis]
        self.sizes = [len(c[0]) for c in clusters]
        self.n = sum(self.sizes)
        self.M = len(clusters)

    def unpack(self, X):
        n, M = self.n, self.M
        z = [mp.mpc(X[2*k], X[2*k + 1]) for k in range(n)]
        Lam = mp.mpc(X[2*n], X[2*n + 1])
        G = []
        for C, (g0, _) in enumerate(self.cl):
            gc = [mp.mpf(v) for v in g0[:-1]] + [X[2*n + 2 + C]]
            G += [self.gam*v for v in gc]
        return z, Lam, G

    def members(self, C):
        s = sum(self.sizes[:C])
        return list(range(s, s + self.sizes[C]))

    def cluster_stats(self, z, G, C):
        mem = self.members(C)
        Zc = mp.fsum(z[k] for k in mem)/len(mem)
        dip = mp.fsum(G[k]*(z[k] - Zc) for k in mem)
        S2 = mp.fsum(G[k]**2 for k in mem)
        return Zc, -2*dip/(S2*Zc), mp.fsum(G[k] for k in mem), S2

    def F(self, X, y):
        z, Lam, G = self.unpack(X)
        zs = [mp.mpc(0)] + z
        Gs = [mp.mpf(1)] + G
        zc = mp.fsum(a*b for a, b in zip(Gs, zs))/mp.fsum(Gs)
        out = []
        for j in range(1, len(zs)):
            s = mp.fsum(Gs[k]/(zs[j] - zs[k]) for k in range(len(zs)) if k != j)
            r = s - Lam*mp.conj(zs[j] - zc)
            out += [r.real, r.imag]
        Z1, q1, _, _ = self.cluster_stats(z, G, 0)
        out += [Z1.real - 1, Z1.imag]
        for C in range(1, self.M):
            out.append(mp.im(self.cluster_stats(z, G, C)[0]*mp.exp(-1j*self.phis[C])))
        out.append(q1.imag - y)
        return out

    def guess(self, y):
        """Leading order plus the O(gamma) relations of the binding cluster (index 0)."""
        es = [self.gam*e_coef(g0, z0) for g0, z0 in self.cl]
        e1 = es[0]
        P = ((3 + e1)*(1 - e1)/4 + y**2)/(2*y*(1 + e1))
        qs, Ls = [], []
        for C, e in enumerate(es):
            if C == 0:
                yc = y
            else:
                disc = P**2*(1 + e)**2 - (3 + e)*(1 - e)/4
                yc = P*(1 + e) + mp.sqrt(disc) if disc > 0 else P*(1 + e)
            q = mp.mpc((1 + e)/2, yc)
            qs.append(q)
            Ls.append((q + 1)/(q - e))
        X, glast = [], []
        for C, (g0, zeta0) in enumerate(self.cl):
            r = mp.sqrt(abs(Ls[C])/abs(Ls[0]))
            ZC = r*mp.exp(1j*self.phis[C]) if C else mp.mpc(1)
            T = (1 + es[C]*Ls[C])/(qs[C]*ZC)
            a = w_internal(zeta0, g0, 0)/T
            cen = mp.fsum(zeta0)/len(zeta0)
            for k in range(len(g0)):
                zk = ZC + self.gam*a*(zeta0[k] - cen)
                X += [zk.real, zk.imag]
            glast.append(mp.mpf(g0[-1]) + self.gam*mp.fsum(v**2 for v in g0)/2)
        return X + [Ls[0].real, Ls[0].imag] + glast


def newton(fam, X, y, iters=40, tol=mp.mpf(10)**-44):
    X = [mp.mpf(v) for v in X]
    h = mp.mpf(10)**-24
    for it in range(iters):
        F0 = fam.F(X, y)
        nrm = max(abs(v) for v in F0)
        if nrm < tol:
            return X, nrm
        J = mp.matrix(len(F0), len(X))
        for i in range(len(X)):
            Xp = list(X)
            Xp[i] += h
            Fp = fam.F(Xp, y)
            for r in range(len(F0)):
                J[r, i] = (Fp[r] - F0[r])/h
        dx = mp.lu_solve(J, mp.matrix(F0))
        X = [X[i] - dx[i] for i in range(len(X))]
    F0 = fam.F(X, y)
    return X, max(abs(v) for v in F0)


def biot_savart(fam, X):
    """Independent check: dz/dt from the Euler law for ALL vortices, the strong one included; kappa from the vortex
    farthest from the centre of vorticity; residual max_j |dz_j/dt - kappa (z_j - zc)|/max|dz/dt|."""
    z, Lam, G = fam.unpack(X)
    zs = [mp.mpc(0)] + z
    Gs = [mp.mpf(1)] + G
    zc = mp.fsum(a*b for a, b in zip(Gs, zs))/mp.fsum(Gs)
    vel = [mp.conj(mp.fsum(Gs[k]/(zs[j] - zs[k]) for k in range(len(zs)) if k != j)/(2j*mp.pi)) for j in range(len(zs))]
    jm = max(range(len(zs)), key=lambda j: abs(zs[j] - zc))
    kap = vel[jm]/(zs[jm] - zc)
    res = max(abs(vel[j] - kap*(zs[j] - zc)) for j in range(len(zs)))/max(abs(v) for v in vel)
    imp = mp.fsum(Gs[j]*abs(zs[j] - zc)**2 for j in range(len(zs)))
    return kap, res, imp, zs, Gs


def P_of(kap):
    return abs(kap.imag)/(-2*kap.real)


def accept(fam, X, nrm):
    kap, res, imp, zs, Gs = biot_savart(fam, X)
    return nrm < mp.mpf(10)**-40 and res < mp.mpf(10)**-40 and abs(imp) < mp.mpf(10)**-40 and kap.real < 0, kap, res, imp


def golden_min(fam, ya, yb, tol=mp.mpf(10)**-8):
    gr = (mp.sqrt(5) - 1)/2
    cache = {}
    last = [None]

    def Pv(yv):
        key = mp.nstr(yv, 25)
        if key not in cache:
            X0 = fam.guess(yv) if last[0] is None else last[0]
            X, nrm = newton(fam, X0, yv)
            ok, kap, res, imp = accept(fam, X, nrm)
            if not ok:
                X, nrm = newton(fam, fam.guess(yv), yv)
                ok, kap, res, imp = accept(fam, X, nrm)
            if not ok:
                raise RuntimeError('not converged at y=%s' % mp.nstr(yv, 8))
            last[0] = X
            cache[key] = (P_of(kap), X)
        return cache[key]
    a, b = mp.mpf(ya), mp.mpf(yb)
    c = b - gr*(b - a)
    d = a + gr*(b - a)
    while b - a > tol:
        if Pv(c)[0] < Pv(d)[0]:
            b = d
        else:
            a = c
        c = b - gr*(b - a)
        d = a + gr*(b - a)
    ym = (a + b)/2
    Pm, X = Pv(ym)
    return ym, Pm, X


def continue_to(fam, X, y0, y1, steps=6):
    """Continuation in y from a solution at y0 to y1."""
    for i in range(1, steps + 1):
        yv = y0 + (y1 - y0)*mp.mpf(i)/steps
        X, nrm = newton(fam, X, yv)
    return X, nrm


def diagnostics(fam, X):
    z, Lam, G = fam.unpack(X)
    out = []
    for C in range(fam.M):
        Zc, q, net, S2 = fam.cluster_stats(z, G, C)
        mem = fam.members(C)
        zeta = [(z[k] - Zc)/fam.gam for k in mem]
        g = [G[k]/fam.gam for k in mem]
        ws = [w_internal(zeta, g, i) for i in range(len(mem))]
        wm = mp.fsum(ws)/len(ws)
        out.append(dict(Z=Zc, q=q, nuhat=net/(S2/2), tre=max(abs(v - wm) for v in ws)/abs(wm),
                        Ppred=(q.imag**2 + mp.mpf(3)/4)/(2*q.imag)))
    return out


def converges(name, vals, order_note, lo=None, hi=mp.mpf('0.2'), last_only=False):
    """vals: list over gamma = 1e-2, 1e-3, 1e-4 of a nonnegative error. Pass if each (or the last) decade shrinks it
    by the factor in [lo, hi]."""
    rat = [vals[i + 1]/vals[i] if vals[i] != 0 else mp.mpf(0) for i in range(len(vals) - 1)]
    use = rat[-1:] if last_only else rat
    ok = all((r <= hi) and (lo is None or r >= lo) for r in use)
    check('%s: %s -> ratios per decade %s (%s)' % (name, ', '.join(mp.nstr(v, 3) for v in vals), ', '.join(mp.nstr(r, 3) for r in rat), order_note), ok)


SQ3 = mp.sqrt(3)/2
GAMMAS = ('1e-2', '1e-3', '1e-4')
stored = []

say('Exact collapses of a strong vortex with weak tight clusters (mpmath %d digits)%s' % (mp.mp.dps, '  [--quick]' if QUICK else ''))
say('\n0. Cluster shapes: translating relative equilibria with zero net circulation')
g3 = [mp.mpf(1), mp.mpf('0.6'), mp.mpf('-1.6')]
z3, w3 = find_TRE(g3, [mp.mpc(0.5, 0.8)])
check('triple (1, 0.6, -1.6): TRE with zero net circulation, equilateral (sides %s, %s, %s)' %
      tuple(mp.nstr(abs(a - b), 12) for a, b in ((z3[1], z3[0]), (z3[2], z3[0]), (z3[2], z3[1]))),
      max(abs(w - w3[0]) for w in w3) < mp.mpf(10)**-40 and abs(abs(z3[2] - z3[0]) - 1) < mp.mpf(10)**-40)
g3b = [mp.mpf('0.8'), mp.mpf('-1.3'), mp.mpf('0.5')]
z3b, w3b = find_TRE(g3b, [mp.mpc(0.4, -0.7)])
check('triple (0.8, -1.3, 0.5): TRE found', max(abs(w - w3b[0]) for w in w3b) < mp.mpf(10)**-40)


def quad_tre(g, seeds):
    for seed in seeds:
        try:
            zz, ww = find_TRE(g, seed)
            dmin = min(abs(zz[i] - zz[j]) for i in range(4) for j in range(i))
            if max(abs(w - ww[0]) for w in ww) < mp.mpf(10)**-40 and dmin > 0.05 and max(abs(v) for v in zz) < 20:
                return zz
        except Exception:
            pass
    return None


g4 = [mp.mpf(1), mp.mpf('0.7'), mp.mpf('-0.5'), mp.mpf('-1.2')]
z4 = quad_tre(g4, ([mp.mpc(0.3, 1.1), mp.mpc(1.2, 0.9)], [mp.mpc(-0.5, 0.8), mp.mpc(0.6, -0.9)], [mp.mpc(0.5, 1.5), mp.mpc(1.5, 0.3)]))
check('quadruple (1, 0.7, -0.5, -1.2): TRE found by Newton from a seed', z4 is not None,
      'zeta = ' + (', '.join(mp.nstr(v, 6) for v in z4) if z4 else 'none'))
g4b = [mp.mpf(1), mp.mpf('0.9'), mp.mpf('0.8'), mp.mpf('-2.7')]
z4b = quad_tre(g4b, ([mp.mpc(0.5, 0.9), mp.mpc(0.5, 0.3)], [mp.mpc(0.3, 1.1), mp.mpc(1.2, 0.9)]))
check('quadruple (1, 0.9, 0.8, -2.7): TRE found', z4b is not None, 'zeta = ' + (', '.join(mp.nstr(v, 6) for v in z4b) if z4b else 'none'))
g2 = [mp.mpf('0.9'), mp.mpf('-0.9')]
z2 = [mp.mpc(0), mp.mpc(1)]

cases = [
    ('pair', [(g2, z2)], [0]),
    ('triple (+,+,-)', [(g3, z3)], [0]),
    ('triple (+,-,+)', [(g3b, z3b)], [0]),
    ('quadruple (+,+,-,-)', [(g4, z4)], [0]),
    ('quadruple (+,+,+,-)', [(g4b, z4b)], [0]),
    ('pair + triple (+,+,-)', [(g2, z2), (g3, z3)], [0, 2.2]),
    ('triple (+,+,-) + quadruple (+,+,-,-) + triple (+,-,+)', [(g3, z3), (g4, z4), (g3b, z3b)], [0, 2.0, 4.1]),
]
if QUICK:
    cases = [c for c in cases if not c[0].startswith('quadruple (+,+,+') and not c[0].startswith('triple (+,+,-) +')]

say('\n1. Cluster arrangements: first-order formula, part (b) structure, part (a)')
for name, cls, phis in cases:
    if any(c[1] is None for c in cls):
        check('%s: cluster shapes available' % name, False)
        continue
    es = [e_coef(g, z) for g, z in cls]
    order = sorted(range(len(cls)), key=lambda C: es[C])
    cls = [cls[C] for C in order]
    es = [es[C] for C in order]
    phis = [0] + [phis[C] - phis[order[0]] for C in order[1:]]
    slope = -2/mp.sqrt(3)*es[0]
    say('\n--- %s; e_C/gamma = %s; predicted (P_min - sqrt3/2)/gamma -> %s  (%.0f s)' %
        (name, [mp.nstr(e, 10) for e in es], mp.nstr(slope, 12), time.time() - T0))
    err, devs, awayP, Pmins = [], [], [], []
    ok_exact = True
    for gam in GAMMAS:
        fam = Family(gam, cls, phis)
        ym, Pm, X = golden_min(fam, mp.mpf('0.65'), mp.mpf('1.1'))
        ok, kap, res, imp = accept(fam, X, mp.mpf(0))
        ok_exact = ok_exact and ok
        G = mp.mpf(gam)
        sl = (Pm - SQ3)/G
        err.append(abs(sl - slope))
        Pmins.append(Pm)
        dg = diagnostics(fam, X)
        devs.append([max(abs(d['nuhat'] - 1) for d in dg), max(abs(d['q'].real - mp.mpf(1)/2) for d in dg),
                     max(d['tre'] for d in dg), max(abs(Pm - d['Ppred']) for d in dg)])
        say('   gamma=%s: P_min = %s at y = %s, (P_min - sqrt3/2)/gamma = %s, residual %s, impulse %s, y_C = %s, |Z_C| = %s' %
            (gam, mp.nstr(Pm, 20), mp.nstr(ym, 10), mp.nstr(sl, 12), mp.nstr(res, 3), mp.nstr(imp, 3),
             [mp.nstr(d['q'].imag, 8) for d in dg], [mp.nstr(abs(d['Z']), 8) for d in dg]))
        z, Lam, Gw = fam.unpack(X)
        stored.append(dict(case=name, gamma=gam, P=mp.nstr(Pm, 40), y=mp.nstr(ym, 20),
                           circulations=[mp.nstr(v, 40) for v in [mp.mpf(1)] + Gw],
                           positions=[[mp.nstr(v.real, 40), mp.nstr(v.imag, 40)] for v in [mp.mpc(0)] + z],
                           clusters=[fam.members(C) for C in range(fam.M)]))
        aw = []
        for yt in ('0.7', '1.05'):
            Xy, nrm = continue_to(fam, X, ym, mp.mpf(yt))
            oky, kapy, resy, impy = accept(fam, Xy, nrm)
            if not oky:
                aw.append(mp.mpf(1))
                continue
            dgy = diagnostics(fam, Xy)
            Py = P_of(kapy)
            aw.append(max(abs(Py - d['Ppred']) for d in dgy) if all(d['q'].imag > 0 for d in dgy) else mp.mpf(1))
        awayP.append(aw)
    check('%s: every point exact (Biot-Savart residual < 1e-40, angular impulse < 1e-40, Re kappa < 0)' % name, ok_exact)
    converges('%s: error of the first-order slope |(P_min - sqrt3/2)/gamma + (2/sqrt3) min e_C|' % name, err,
              'O(gamma): ratio in [0.05, 0.2]', lo=mp.mpf('0.05'), hi=mp.mpf('0.2'))
    if es[0] == 0:
        check('%s: e = 0 (a pair binds): P_min > sqrt3/2 at every gamma, (P_min - sqrt3/2)/gamma^2 = %s' %
              (name, ', '.join(mp.nstr((P - SQ3)/mp.mpf(g)**2, 8) for P, g in zip(Pmins, GAMMAS))), all(P > SQ3 for P in Pmins))
    else:
        check('%s: e > 0: P_min < sqrt3/2 at gamma = 1e-3 and 1e-4 (no analogue of the fixed-circulation bound for pairs)' % name,
              Pmins[1] < SQ3 and Pmins[2] < SQ3, ', '.join(mp.nstr(P - SQ3, 6) for P in Pmins))
    labels = ['|nuhat - 1|', '|Re q - 1/2|', 'TRE defect', '|P - (y^2+3/4)/(2y)|']
    for i, lab in enumerate(labels):
        converges('%s: part (b) at the minimum, max over clusters of %s' % (name, lab), [d[i] for d in devs],
                  'at least linear in gamma over the last decade', last_only=True)
    for j, yt in enumerate(('0.7', '1.05')):
        converges('%s: part (b) at y_1 = %s, max_C |P - (y_C^2+3/4)/(2 y_C)| with every y_C > 0' % (name, yt),
                  [a[j] for a in awayP], 'at least linear in gamma over the last decade', last_only=True)

say('\n2. A (+,-,-) triple and the closed form for equilateral triples  (%.0f s)' % (time.time() - T0))
for gs in ([mp.mpf(-1), mp.mpf('-0.6'), mp.mpf('1.6')], [mp.mpf(1), mp.mpf('0.6'), mp.mpf('-1.6')], [mp.mpf('0.3'), mp.mpf('-1.1'), mp.mpf('0.8')]):
    z, w = find_TRE(gs, [mp.mpc(0.5, 0.8)])
    S2 = sum(v**2 for v in gs)
    closed = 2*mp.sqrt(3)*gs[0]*gs[1]*gs[2]/S2
    pred = -2/mp.sqrt(3)*e_coef(gs, z)
    check('equilateral triple %s: slope -(2/sqrt3) e/gamma = %s equals 2 sqrt3 g1 g2 g3/S2 = %s' %
          ([mp.nstr(v, 3) for v in gs], mp.nstr(pred, 12), mp.nstr(closed, 12)), abs(pred - closed) < mp.mpf(10)**-40)
g = [mp.mpf(-1), mp.mpf('-0.6'), mp.mpf('1.6')]
z, w = find_TRE(g, [mp.mpc(0.5, 0.8)])
pred = -2/mp.sqrt(3)*e_coef(g, z)
err, Pmins = [], []
for gam in GAMMAS:
    fam = Family(gam, [(g, z)], [0])
    ym, Pm, X = golden_min(fam, mp.mpf('0.65'), mp.mpf('1.1'))
    ok, kap, res, imp = accept(fam, X, mp.mpf(0))
    err.append(abs((Pm - SQ3)/mp.mpf(gam) - pred))
    Pmins.append(Pm)
    say('   (+,-,-) gamma=%s: P_min = %s, slope %s, predicted %s, residual %s' % (gam, mp.nstr(Pm, 16), mp.nstr((Pm - SQ3)/mp.mpf(gam), 10), mp.nstr(pred, 10), mp.nstr(res, 3)))
check('(+,-,-) triple beside a positive strong vortex (e < 0): P_min > sqrt3/2 at every gamma', all(P > SQ3 for P in Pmins))
converges('(+,-,-) triple: error of the first-order slope', err, 'O(gamma): ratio in [0.05, 0.2]', lo=mp.mpf('0.05'))

say('\n3. Part (d): singletons (rings with a strong centre at fixed relative rotation)  (%.0f s)' % (time.time() - T0))
for n, th in ((3, mp.mpf('0.5')), (2, mp.mpf('0.9')), (5, mp.mpf('0.3'))):
    eps = mp.exp(2j*mp.pi/n)
    errs = []
    for G0 in (mp.mpf(100), mp.mpf(1000), mp.mpf(10000)):
        A, B, C = n - 1, -2*(n - G0), (n - 1) - 2*G0
        x = (-B + mp.sqrt(B**2 - 4*A*C))/(2*A)
        gam = 1/G0
        zs = [mp.mpc(0)] + [eps**k for k in range(n)] + [mp.sqrt(x)*mp.exp(1j*th)*eps**k for k in range(n)]
        Gs = [mp.mpf(1)] + [gam*x]*n + [-gam]*n
        zc = mp.fsum(a*b for a, b in zip(Gs, zs))/mp.fsum(Gs)
        vel = [mp.conj(mp.fsum(Gs[k]/(zs[j] - zs[k]) for k in range(len(zs)) if k != j)/(2j*mp.pi)) for j in range(len(zs))]
        kap = vel[1]/(zs[1] - zc)
        res = max(abs(vel[j] - kap*(zs[j] - zc)) for j in range(len(zs)))/max(abs(v) for v in vel)
        P = P_of(kap)
        weak = list(range(1, 2*n + 1))
        dmin = min(abs(zs[i] - zs[j]) for i in weak for j in weak if i < j)
        c = min(dmin, 1/x, 1/mp.sqrt(x), mp.mpf(1))
        nw = 2*n
        bound = c**3/(40*nw*gam) if gam <= c**3/(20*nw) else None
        lim = mp.tan(n*th/2)/n
        errs.append(abs(P*gam - lim))
        check('ring n=%d theta=%s Gamma0=%d: exact collapse (residual %s, Re kappa %s); P = %s; P >= c^3/(40 n gamma) = %s (c = %s)' %
              (n, th, int(G0), mp.nstr(res, 3), mp.nstr(kap.real, 3), mp.nstr(P, 10), mp.nstr(bound, 5) if bound else 'n/a, gamma > c^3/(20 n)', mp.nstr(c, 4)),
              res < mp.mpf(10)**-40 and kap.real < 0 and (bound is None or P >= bound))
    converges('ring n=%d theta=%s: |P gamma - tan(n theta/2)/n|' % (n, th), errs, 'O(gamma): ratio in [0.05, 0.2]', lo=mp.mpf('0.05'))

say('\n4. Controls  (%.0f s)' % (time.time() - T0))
for f in ('collapse-euler-n33.json', 'collapse-euler-n61.json', 'collapse-euler-n603.json'):
    d = json.load(open(os.path.join(DATA, f)))
    G = [mp.mpf(v) for v in d['G']]
    Nv = len(G)
    idx = sorted(range(Nv), key=lambda i: -abs(G[i]))
    top, second = abs(G[idx[0]]), abs(G[idx[1]])
    weak = [i for i in range(Nv) if abs(G[i]) < top/2]
    wsum = mp.fsum(G[i] for i in weak)/top
    pos = sum(1 for i in weak if G[i] > 0)
    P = mp.mpf(d['P'])
    check('two-arm minimizer %s: P = %s < sqrt3/2, outside the regime of Theorem 1: two co-dominant vortices (|Gamma| ratio %s), so '
          'a class K(n, c) contains it only with gamma >= %s c; the %d weaker vortices carry net circulation %s times the strongest and '
          'all %d have one sign, so none of them forms a cluster of net circulation O(gamma^2)' %
          (f, mp.nstr(P, 10), mp.nstr(second/top, 6), mp.nstr(second/top, 3), len(weak), mp.nstr(wsum, 5), pos),
          P < SQ3 and second/top > mp.mpf('0.99') and wsum > 1 and pos == len(weak))
# The seven-vortex collapse of Demina and Kudryashov (TCFD 28 (2014), Table 1, Fig. 1a)
c2p = mp.mpf(13)/18
phi = mp.acos(c2p)/2
zs = [mp.mpc(0), mp.mpc(2), mp.mpc(-2), 2*mp.expj(phi), -2*mp.expj(phi), mp.mpc(mp.mpf(4)/3), mp.mpc(-mp.mpf(4)/3)]
Gs = [mp.mpf(6383)/2250, mp.mpf(14)/15, mp.mpf(14)/15, -mp.mpf(62)/45, -mp.mpf(62)/45, mp.mpf(1), mp.mpf(1)]
vel = [mp.conj(mp.fsum(Gs[k]/(zs[j] - zs[k]) for k in range(7) if k != j)/(2j*mp.pi)) for j in range(7)]
kap = vel[1]/zs[1]
res = max(abs(vel[j] - kap*zs[j]) for j in range(7))/max(abs(v) for v in vel)
P7 = P_of(kap)
check('seven-vortex collapse of Demina and Kudryashov: self-similar (residual %s), P = %s = 12433/(1240 sqrt155)' % (mp.nstr(res, 3), mp.nstr(P7, 12)),
      res < mp.mpf(10)**-40 and abs(P7 - mp.mpf(12433)/(1240*mp.sqrt(155))) < mp.mpf(10)**-40)
g7 = [v/Gs[0] for v in Gs[1:]]
inK = []
for c in (mp.mpf(1)/3, mp.mpf(1)/10, mp.mpf(1)/100):
    gam = c
    gg = [v/gam for v in g7]
    Zs = zs[1:]
    ok = all(c <= abs(v) <= 1/c for v in gg) and all(c <= abs(Zz) <= 1/c for Zz in Zs) and \
        all(abs(Zs[i] - Zs[j]) >= c for i in range(6) for j in range(i))
    inK.append(ok)
check('   it lies in K(6, c) with every weak vortex a singleton and gamma = c, for c = 1/3, 1/10, 1/100, with P < sqrt3/2: '
      'so gamma_1(6, c, delta) <= c for delta < sqrt3/2 - P, not uniform as c -> 0', all(inK) and P7 < SQ3)
d4 = json.load(open(os.path.join(DATA, 'collapse-euler-n4-minimum.json')))
G4 = [mp.mpf(v) for v in d4['G']]
Z4 = [mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in d4['z']]
js = max(range(4), key=lambda i: abs(G4[i]))
gw = [G4[i]/G4[js] for i in range(4) if i != js]
zw = [Z4[i] - Z4[js] for i in range(4) if i != js]
Zc4 = mp.fsum(zw)/3
S2w = mp.fsum(v**2 for v in gw)
nuhat4 = mp.fsum(gw)/(S2w/2)
signs = ''.join('+' if v > 0 else '-' for v in sorted(gw, key=lambda v: -v))
spread = max(abs(a - b) for a in zw for b in zw)/abs(Zc4)
check('certified four-vortex minimum of the companion paper (P_4 = %s): one strong vortex and a (%s) triple of ratios %s, '
      'net circulation / ((1/2) sum Gamma_k^2) = %s, triple diameter / distance = %s, first-order slope sign %s' %
      (mp.nstr(mp.mpf(d4['P']), 10), ','.join(signs), [mp.nstr(v, 4) for v in gw], mp.nstr(nuhat4, 5), mp.nstr(spread, 3),
       '-' if gw[0]*gw[1]*gw[2] < 0 else '+'),
      signs == '++-' and abs(nuhat4 - 1) < mp.mpf('0.1') and gw[0]*gw[1]*gw[2] < 0 and mp.mpf(d4['P']) < SQ3)

if not QUICK:
    json.dump(dict(about='Exact self-similar collapses of a strong vortex (circulation 1 at the origin) with weak tight clusters, '
                   'computed by verify_cluster_collapses.py at 50 digits; values to 40 digits. P is the least value along the '
                   'family with the cluster shapes and directions fixed.', collapses=stored),
              open(os.path.join(DATA, 'cluster-collapses.json'), 'w'), indent=1)
say('\n%d checks, %d failed  (%.0f s)' % (NCHK[0], len(FAILED), time.time() - T0))
with open(os.path.join(DATA, 'verify-cluster-collapses%s.txt' % ('-quick' if QUICK else '')), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
