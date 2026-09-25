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
"""The continuum model of Section 6 of paper/collapse-without-rotation.tex, in binary64: a C2-symmetric (point
symmetric) vortex sheet z(x), -1 <= x <= 1, through the collision point, with |dz/dx| = L (x proportional to arc
length) and circulation g(x) dx, g(x) = sqrt(1 - x^2) h(x), together with two point vortices of circulation -1 at
+-z_n. Self-similar collapse: the principal-value Birkhoff-Rott integral equals Lambda conj(z) on the sheet, and the
velocity at z_n equals Lambda conj(z_n), where Lambda = 2 pi i conj(kappa); P = -Re Lambda/(2 Im Lambda). Gauge: the
tip z(1) = 1.

Discretization: theta (the tangent angle) and h as even Chebyshev series of m terms each (T_0, T_2, ..., T_{2m-2});
the principal-value integral by Gauss-Chebyshev quadrature of the second kind with n nodes t_k, collocated at the
positive zeros x_r of T_{n+1} (the odd symmetry makes the negative ones redundant); z by Chebyshev integration of
e^{i theta} on K + 1 points. Unknowns u = [a (m), b (m), L, Re z_n, Im z_n, Re Lambda, Im Lambda]. The discrete system
has a one-dimensional solution set (the continuum family); minimise() minimizes P along it.
Class Model and gn(): the model and Gauss-Newton; nullvec(), solve_plane(), minimise(): P along the family.
"""
import numpy as np
from numpy.polynomial import chebyshev as C

class Model:
    def __init__(s, m, n, K=None, kind=2):
        s.m, s.n = m, n
        s.K = K or max(4*m, 64)
        if kind == 2:   # g = sqrt(1-x^2) h : density vanishing like sqrt at the tips
            k = np.arange(1, n+1); s.t = np.cos(k*np.pi/(n+1)); s.w = np.pi/(n+1)*np.sin(k*np.pi/(n+1))**2
            r = np.arange(1, n+2); x = np.cos((2*r-1)*np.pi/(2*(n+1)))
        else:           # g = h/sqrt(1-x^2) : inverse-sqrt class (contains the sqrt class as h(+-1) = 0)
            k = np.arange(1, n+1); s.t = np.cos((2*k-1)*np.pi/(2*n)); s.w = np.full(n, np.pi/n)
            r = np.arange(1, n); x = np.cos(r*np.pi/n)
        s.x = x[x > 1e-12]                       # positive collocation points (F is odd)
        # Chebyshev points for building z
        j = np.arange(s.K+1); s.cp = np.cos(j*np.pi/s.K)
        s.nv = 2*m + 5
        # precomputed linear maps: a -> theta(cp); values at cp -> z/L at t, x, 1 ; b -> h(t)
        Vcp = C.chebvander(s.cp, 2*m-2)[:, ::2]
        s.Vcp = Vcp
        Vfit = C.chebvander(s.cp, s.K)                      # square (K+1)x(K+1)
        Finv = np.linalg.inv(Vfit)                          # values -> coefficients
        Iop = np.zeros((s.K+2, s.K+1))
        for j in range(s.K+1):
            e = np.zeros(s.K+1); e[j] = 1; Iop[:, j] = C.chebint(e, lbnd=0)
        def ev(pts): return C.chebvander(np.atleast_1d(pts), s.K+1) @ Iop @ Finv
        s.Et, s.Ex, s.E1 = ev(s.t), ev(s.x), ev(1.0)[0]
        s.Vt = C.chebvander(s.t, 2*m-2)[:, ::2]
    def coef(s, a):
        c = np.zeros(2*len(a)-1); c[::2] = a; return c
    def unpack(s, u):
        m = s.m
        a = u[:m]; b = u[m:2*m]; L = u[2*m]; zn = u[2*m+1] + 1j*u[2*m+2]; lam = u[2*m+3] + 1j*u[2*m+4]
        return a, b, L, zn, lam
    def zfun(s, a, L):
        th = C.chebval(s.cp, s.coef(a))
        e = np.exp(1j*th)
        ce = C.chebfit(s.cp, e.real, s.K) + 1j*C.chebfit(s.cp, e.imag, s.K)
        cz = C.chebint(ce, lbnd=0)*L
        return cz
    def fields(s, u):
        a, b, L, zn, lam = s.unpack(u)
        e = np.exp(1j*(s.Vcp @ a))
        zt = L*(s.Et @ e); zx = L*(s.Ex @ e); z1 = L*(s.E1 @ e)
        Gk = s.w*(s.Vt @ b)                        # point-vortex representation of the sheet
        return a, b, L, zn, lam, zt, zx, z1, Gk
    def F(s, u):
        a, b, L, zn, lam, zt, zx, z1, Gk = s.fields(u)
        ws = (Gk[None, :]/(zx[:, None] - zt[None, :])).sum(1)
        Fs = ws - 1/(zx - zn) - 1/(zx + zn) - lam*np.conj(zx)
        wn = (Gk/(zn - zt)).sum() - 1/(2*zn)
        Fn = wn - lam*np.conj(zn)
        return np.r_[Fs.real, Fs.imag, Fn.real, Fn.imag, z1.real - 1, z1.imag]
    def J(s, u, h=1e-7):
        f0 = s.F(u); Jm = np.zeros((len(f0), len(u)))
        for i in range(len(u)):
            e = np.zeros(len(u)); d = h*max(1.0, abs(u[i])); e[i] = d
            Jm[:, i] = (s.F(u+e) - s.F(u-e))/(2*d)
        return f0, Jm
    def P(s, u):
        lam = u[2*s.m+3] + 1j*u[2*s.m+4]
        return -lam.real/(2*lam.imag)
    def diag(s, u):
        a, b, L, zn, lam, zt, zx, z1, Gk = s.fields(u)
        Gs = Gk.sum()
        I = (Gk*np.abs(zt)**2).sum() - 2*abs(zn)**2
        return dict(Gsheet=Gs, Gsheet_minus=Gs-2-np.sqrt(2), I=I, P=s.P(u), L=L, zn=zn, lam=lam,
                    h0=C.chebval(0, s.coef(b)), gam0=C.chebval(0, s.coef(b))/L)
    def resize(s, u, m2):
        a, b, L, zn, lam = s.unpack(u)
        a2 = np.zeros(m2); b2 = np.zeros(m2); k = min(m2, s.m)
        a2[:k] = a[:k]; b2[:k] = b[:k]
        return np.r_[a2, b2, L, zn.real, zn.imag, lam.real, lam.imag]

def gn(M, u, iters=50, tol=1e-14, verbose=False):
    for it in range(iters):
        f, Jm = M.J(u)
        r = np.linalg.norm(f)
        if verbose: print(it, r, M.P(u))
        if r < tol: break
        du = np.linalg.lstsq(Jm, -f, rcond=None)[0]
        t = 1.0
        while t > 1e-6:
            un = u + t*du
            if np.linalg.norm(M.F(un)) < (1 - 1e-4*t)*r: break
            t /= 2
        u = un
    return u, np.linalg.norm(M.F(u))


def nullvec(M, u):
    f, Jm = M.J(u); U, s, Vt = np.linalg.svd(Jm); return Vt[-1], s

def solve_plane(M, u, n0, c0, iters=30, tol=1e-13):
    """Gauss-Newton on [F(u); n0.u - c0] (full column rank)."""
    for it in range(iters):
        f, Jm = M.J(u)
        fa = np.r_[f, n0 @ u - c0]; Ja = np.vstack([Jm, n0])
        r = np.linalg.norm(fa)
        if r < tol: break
        du = np.linalg.lstsq(Ja, -fa, rcond=None)[0]
        u = u + du
        if np.linalg.norm(du) < 1e-15*np.linalg.norm(u): break
    return u, np.linalg.norm(np.r_[M.F(u), n0 @ u - c0])

def minimise(M, u, h=0.004, verbose=True, rounds=12):
    for rnd in range(rounds):
        n0, s = nullvec(M, u)
        c = n0 @ u
        pts = []
        for dc in (-h, 0.0, h):
            w, r = solve_plane(M, u.copy() + dc*n0, n0, c + dc)
            pts.append((dc, M.P(w), r, w))
        (x1, p1, r1, w1), (x2, p2, r2, w2), (x3, p3, r3, w3) = pts
        d1 = (p3 - p1)/(2*h); d2 = (p3 - 2*p2 + p1)/h**2
        step = -d1/d2 if d2 > 0 else -np.sign(d1)*4*h
        step = np.clip(step, -8*h, 8*h)
        u, r = solve_plane(M, w2 + step*n0, n0, c + step)
        if verbose: print('round %d  P %.14f  dP/dc %.3e  d2P %.4e  step %.3e  resid %.1e  sv %.1e %.1e' % (
            rnd, M.P(u), d1, d2, step, r, s[-1]/s[0], s[-2]/s[0]), flush=True)
        if abs(step) < 1e-7: break
        h = max(min(h, 2*abs(step)), 1e-4)
    return u, d2

