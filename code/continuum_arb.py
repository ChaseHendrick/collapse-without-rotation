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
"""The continuum model of continuum_model.py evaluated in ball arithmetic (FLINT/Arb through python-flint): the same
collocation residual, with every constant (nodes, weights, the Chebyshev transform and integration matrices)
computed in Arb at the working precision; and a mixed-precision refinement (binary64 Jacobian, Arb residuals) that
minimizes P along the one-parameter family by a three-point Newton step in the family coordinate.

These are high-precision numerics, not a computer-assisted proof: the refined solutions solve the DISCRETE
collocation system to about 1e-39, and the continuum problem only up to the truncation error of the discretization,
which verify_continuum_limit.py estimates by comparing resolutions.

Run as a program to refine a stored solution at another resolution:
    python3 code/continuum_arb.py [m n prec [start.json]]
defaults m = 64, n = 384, prec = 192 bits, start data/continuum/continuum-m64-n384.json (rounded to binary64 first,
so the refinement starts from 16 digits). It prints P after each round and writes data/continuum/refined-m<m>-n<n>.json.
At the defaults it takes about a minute; m = 140, n = 840 at 256 bits takes about four minutes.
"""
import os
import sys
import time
import json
import numpy as np
from flint import arb, acb, arb_mat, acb_mat, fmpq, ctx
from continuum_model import Model, nullvec, solve_plane

class HP:
    def __init__(s, m, n, K=None, prec=192):
        ctx.prec = prec
        s.m, s.n = m, n; s.K = K or max(4*m, 64); K = s.K
        pi = arb.pi()
        s.phit = [pi*k/(n+1) for k in range(1, n+1)]
        s.t = [p.cos() for p in s.phit]
        s.w = [pi/(n+1)*p.sin()**2 for p in s.phit]
        ph_x = [pi*(2*r-1)/(2*(n+1)) for r in range(1, n+2)]
        s.phix = [p for p in ph_x if p.cos() > arb('1e-12')]      # positive collocation points (same set as float)
        phcp = [pi*j/K for j in range(K+1)]
        # a -> theta(cp):  T_{2i}(cos phi) = cos(2 i phi)
        s.Vcp = arb_mat([[ (2*i*p).cos() for i in range(m)] for p in phcp])
        # values at cp -> Chebyshev coefficients (DCT-I)
        Fi = []
        for k in range(K+1):
            row = []
            for j in range(K+1):
                c = (phcp[j]*k).cos()*2/K
                if j == 0 or j == K: c = c/2
                if k == 0 or k == K: c = c/2
                row.append(c)
            Fi.append(row)
        s.Finv = arb_mat(Fi)
        # integration with lbnd = 0 : coefficients (K+1) -> (K+2)
        I = [[fmpq(0)]*(K+1) for _ in range(K+2)]
        for k in range(K+1):
            if k == 0: I[1][0] += 1
            elif k == 1: I[2][1] += fmpq(1, 4); I[0][1] += fmpq(-1, 4)   # int T1 = T2/4 (+const)
            else:
                I[k+1][k] += fmpq(1, 2*(k+1)); I[k-1][k] -= fmpq(1, 2*(k-1))
        # constant so that value at 0 vanishes: b0 = -sum_{j>=1} b_j T_j(0)
        Tz = [ [1, 0, -1, 0][j % 4] for j in range(K+2)]
        for k in range(K+1):
            sacc = fmpq(0)
            for j in range(1, K+2): sacc += I[j][k]*Tz[j]
            I[0][k] = -sacc + 0   # overwrite b0 (the chebint constant)
        s.Iop = arb_mat([[arb(I[j][k]) for k in range(K+1)] for j in range(K+2)])
        # evaluation at t, x, 1
        def V(phis): return arb_mat([[ (k*p).cos() for k in range(K+2)] for p in phis])
        s.Vt_z = V(s.phit); s.Vx_z = V(s.phix); s.V1_z = V([arb(0)])
        s.Vt_h = arb_mat([[ (2*i*p).cos() for i in range(m)] for p in s.phit])
        s.nx = len(s.phix)
    def F(s, u):
        """u: list of arb of length 2m+5. returns list of arb residuals (same layout as cont.Model.F)."""
        m = s.m
        a = arb_mat([[x] for x in u[:m]]); b = arb_mat([[x] for x in u[m:2*m]])
        L = u[2*m]; zn = acb(u[2*m+1], u[2*m+2]); lam = acb(u[2*m+3], u[2*m+4])
        th = s.Vcp*a
        e = acb_mat([[acb(0, th[j, 0]).exp()] for j in range(th.nrows())])
        ce = acb_mat(s.Finv)*e
        cz = acb_mat(s.Iop)*ce
        zt = acb_mat(s.Vt_z)*cz; zx = acb_mat(s.Vx_z)*cz; z1 = (acb_mat(s.V1_z)*cz)[0, 0]*L
        zt = [zt[k, 0]*L for k in range(s.n)]; zx = [zx[r, 0]*L for r in range(s.nx)]
        hv = s.Vt_h*b
        Gk = [s.w[k]*hv[k, 0] for k in range(s.n)]
        Fr = []; Fi = []
        for r in range(s.nx):
            zr = zx[r]; acc = acb(0)
            for k in range(s.n): acc += Gk[k]/(zr - zt[k])
            f = acc - 1/(zr - zn) - 1/(zr + zn) - lam*zr.conjugate()
            Fr.append(f.real); Fi.append(f.imag)
        acc = acb(0)
        for k in range(s.n): acc += Gk[k]/(zn - zt[k])
        fn = acc - 1/(2*zn) - lam*zn.conjugate()
        s.last = dict(Gsheet=sum(Gk), I=sum(Gk[k]*(zt[k].real**2 + zt[k].imag**2) for k in range(s.n)) - 2*(zn.real**2 + zn.imag**2))
        return Fr + Fi + [fn.real, fn.imag, z1.real - 1, z1.imag]

def tofloat(v): return np.array([float(x.mid()) for x in v])
def P_of(u, m): return -u[2*m+3]/(2*u[2*m+4])

def refine(H, M, u, n0, c, Jaug_pinv=None, iters=8, verbose=False):
    """Mixed-precision Gauss-Newton on [F(u); n0.u - c] ; u list of arb; n0 float array; c arb."""
    if Jaug_pinv is None:
        f, Jm = M.J(tofloat(u)); Ja = np.vstack([Jm, n0]); Jaug_pinv = np.linalg.pinv(Ja)
    n0a = [arb(float(x)) for x in n0]
    hist = []
    for it in range(iters):
        Fv = H.F(u)
        cres = sum((n0a[i]*u[i] for i in range(len(u))), arb(0)) - c
        fa = np.r_[tofloat(Fv), float(cres.mid())]
        r = np.linalg.norm(fa); hist.append(r)
        if verbose: print('   it %d resid %.3e P %s' % (it, r, P_of(u, H.m).str(30, radius=False)), flush=True)
        du = Jaug_pinv @ fa
        u = [u[i] - arb(float(du[i])) for i in range(len(u))]
        if len(hist) > 2 and hist[-1] > 0.3*hist[-2] and hist[-1] < 1e-25: break
    return u, hist, Jaug_pinv


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(os.path.dirname(HERE), 'data', 'continuum')
    args = sys.argv[1:]
    m = int(args[0]) if len(args) > 0 else 64
    n = int(args[1]) if len(args) > 1 else 384
    prec = int(args[2]) if len(args) > 2 else 192
    src = args[3] if len(args) > 3 else os.path.join(DATA, 'continuum-m64-n384.json')
    d0 = json.load(open(src))
    u = np.array([float(x) for x in d0['u']])
    M0 = Model(d0['m'], d0['n'])
    u = M0.resize(u, m)
    M = Model(m, n)
    t0 = time.time()
    n0, s = nullvec(M, u)
    c0 = float(n0 @ u)
    u, r = solve_plane(M, u, n0, c0)
    f, Jm = M.J(u)
    pinv = np.linalg.pinv(np.vstack([Jm, n0]))
    print('binary64 start: residual %.2e  P %.15f  (%.0f s)' % (r, M.P(u), time.time() - t0), flush=True)
    H = HP(m, n, prec=prec)
    uh = [arb(float(x)) for x in u]
    c = arb(c0)

    def P_at(cc, ustart):
        w, hist, _ = refine(H, M, ustart, n0, cc, Jaug_pinv=pinv, iters=14)
        return P_of(w, m), w, hist
    for rnd, dl in enumerate([1e-6, 1e-9, 1e-12]):
        dl = arb(dl)
        Pm, wm, hm = P_at(c - dl, uh)
        P0, w0, h0 = P_at(c, uh)
        Pp, wp, hp_ = P_at(c + dl, uh)
        d1 = (Pp - Pm)/(2*dl)
        d2 = (Pp - 2*P0 + Pm)/dl**2
        step = -d1/d2
        print('round %d: P %s  dP/dc %.3e  d2P/dc2 %.6f  step %.3e  residual %.1e  (%.0f s)' % (
            rnd, P0.mid().str(45, radius=False), float(d1.mid()), float(d2.mid()), float(step.mid()), h0[-1], time.time() - t0), flush=True)
        c = c + step
        uh = [w0[i] + step*arb(float(n0[i])) for i in range(len(w0))]
    Pf, wf, hf = P_at(c, uh)
    H.F(wf)
    print('final m %d n %d prec %d: P = %s  residual %.2e  sheet circulation - (2 + sqrt2) %.1e  angular impulse %.1e' % (
        m, n, prec, Pf.mid().str(45, radius=False), hf[-1], float((H.last['Gsheet'] - 2 - arb(2).sqrt()).mid()), float(H.last['I'].mid())))
    out = os.path.join(DATA, 'refined-m%d-n%d.json' % (m, n))
    json.dump(dict(m=m, n=n, prec=prec, P=Pf.mid().str(55, radius=False), u=[x.mid().str(55, radius=False) for x in wf], resid=hf[-1]), open(out, 'w'))
    print('wrote', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    main()
