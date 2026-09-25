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
"""Shared numerics for self-similar collapse WITHOUT rotation (kappa real) in the alpha-models, used by
verify_phase_diagram.py, verify_family_structure.py and verify_stability.py.

Law: dz_j/dt = (i/2 pi) sum_k G_k (z_j - z_k)|z_j - z_k|^(-alpha-2)  (alpha = 0 Euler, alpha = 1 SQG).
Self-similar with the collapse point at the origin: dz_j/dt = kappa z_j, in the gauge 2 pi kappa = -1 + i b
(b = 0 on the set Z of collapses without rotation). Equations E_j = 2 pi V_j + (1 - i b) z_j = 0 (2N reals).
Full variable vector u = [x (N), y (N), G (N), alpha, b]; a mask selects the free unknowns. jac_full is the
analytic Jacobian of the equations with respect to u. kkt_polish is Newton's method on the first-order (KKT)
system of min alpha on Z, with the Hessian of the Lagrangian from central differences of J^T lambda, and returns the
eigenvalues of the reduced Hessian (on the tangent space of the constraints).
"""
import json
import numpy as np
from scipy.optimize import minimize

TWO_PI = 2 * np.pi


def vel(z, G, a):
    d = z[:, None] - z[None, :]
    r2 = (d * d.conj()).real
    np.fill_diagonal(r2, 1.0)
    w = d * r2 ** (-(a + 2) / 2)
    np.fill_diagonal(w, 0)
    return 1j / TWO_PI * (w @ G)


def unpack(u, N):
    return u[:N] + 1j * u[N:2 * N], u[2 * N:3 * N], u[3 * N], u[3 * N + 1]


def pack(z, G, a, b):
    return np.concatenate([z.real, z.imag, G, [a, b]])


def eqs(u, N):
    z, G, a, b = unpack(u, N)
    E = TWO_PI * vel(z, G, a) + (1 - 1j * b) * z
    return np.concatenate([E.real, E.imag])


def jac_full(u, N):
    """Jacobian of eqs w.r.t. the full vector u (2N x (3N+2))."""
    z, G, a, b = unpack(u, N)
    d = z[:, None] - z[None, :]
    r2 = (d * d.conj()).real
    np.fill_diagonal(r2, 1.0)
    rp = r2 ** (-(a + 2) / 2)
    rp4 = rp / r2
    np.fill_diagonal(rp, 0)
    np.fill_diagonal(rp4, 0)
    w = d * rp
    dwdx = rp - (a + 2) * d.real * d * rp4
    dwdy = 1j * rp - (a + 2) * d.imag * d * rp4
    # 2 pi V_j = i sum_k G_k w_jk
    Jx = -1j * dwdx * G[None, :]
    Jx[np.diag_indices(N)] = 1j * (dwdx @ G)
    Jy = -1j * dwdy * G[None, :]
    Jy[np.diag_indices(N)] = 1j * (dwdy @ G)
    Jx = Jx + (1 - 1j * b) * np.eye(N)
    Jy = Jy + 1j * (1 - 1j * b) * np.eye(N)
    JG = 1j * w
    lr = 0.5 * np.log(r2)
    Ja = 1j * ((-lr * w) @ G)
    Jb = -1j * z
    Jc = np.hstack([Jx, Jy, JG, Ja[:, None], Jb[:, None]])
    return np.vstack([Jc.real, Jc.imag])


class System:
    """Equations restricted to free variables; fixed ones are held at u0."""

    def __init__(self, u0, N, free):
        self.N = N
        self.u0 = np.array(u0, float)
        self.free = np.array(free, bool)
        self.idx = np.where(self.free)[0]

    def full(self, v):
        u = self.u0.copy()
        u[self.idx] = v
        return u

    def F(self, v):
        return eqs(self.full(v), self.N)

    def J(self, v):
        return jac_full(self.full(v), self.N)[:, self.idx]


def normalize(z, G, a, kappa=None):
    """Translate to the collapse point, scale so that Re(2 pi kappa) = -1 (collapse), return (z, G, b)."""
    z = np.asarray(z, complex)
    G = np.asarray(G, float)
    V = vel(z, G, a)
    # least squares V = kappa (z - zc): unknowns kappa, c = -kappa zc
    A = np.stack([z, np.ones_like(z)], 1)
    sol = np.linalg.lstsq(A, V, rcond=None)[0]
    k, c = sol
    zc = -c / k
    z = z - zc
    k2 = TWO_PI * k
    G = G * (-1 / k2.real)           # now Re(2 pi kappa) = -1
    b = (k2 * (-1 / k2.real)).imag
    return z, G, b


def gauge(z, G, ir=None, ig=None):
    """Rotate so vortex ir (farthest) is on the positive real axis; scale lengths so that |G_ig| = 1 keeps kappa.
    Scaling lengths by s multiplies kappa by s^(-a-2); scaling G by c multiplies kappa by c. Keep kappa: c = s^(a+2)."""
    if ir is None:
        ir = int(np.argmax(np.abs(z)))
    z = z * np.exp(-1j * np.angle(z[ir]))
    if ig is None:
        ig = int(np.argmax(np.abs(G)))
    return z, G, ir, ig


def rescale_G1(z, G, a, ig):
    c = 1.0 / abs(G[ig])
    s = c ** (1.0 / (a + 2))
    return z * s, G * c


def free_mask(N, ir, ig, a_free, b_free):
    m = np.ones(3 * N + 2, bool)
    m[N + ir] = False
    m[2 * N + ig] = False
    m[3 * N] = a_free
    m[3 * N + 1] = b_free
    return m


def newton_project(S, v, tol=1e-13, maxit=50, damp=True):
    """Minimum-norm Gauss-Newton onto S.F = 0."""
    for it in range(maxit):
        F = S.F(v)
        nf = np.linalg.norm(F)
        if nf < tol:
            return v, nf, True
        J = S.J(v)
        if not (np.all(np.isfinite(J)) and np.isfinite(nf)):
            return v, np.inf, False
        try:
            step = -J.T @ np.linalg.solve(J @ J.T, F)
        except np.linalg.LinAlgError:
            return v, np.inf, False
        t = 1.0
        while damp and t > 1e-4:
            vn = v + t * step
            if np.linalg.norm(S.F(vn)) < nf:
                break
            t /= 2
        v = v + t * step
    nf = np.linalg.norm(S.F(v))
    return v, nf, nf < tol * 100


def null_basis(J):
    U, s, Vt = np.linalg.svd(J)
    r = np.sum(s > s[0] * 1e-12)
    return Vt[r:].T, s


def kkt_polish(S, v, obj_grad, lam=None, maxit=30, tol=1e-12, h=1e-6, verbose=False):
    """Newton on the KKT system of min obj(v) s.t. F(v)=0 with obj linear (gradient obj_grad constant).
    Hessian of the Lagrangian from central differences of J^T lam."""
    n = len(v)
    J = S.J(v)
    m = J.shape[0]
    bad = dict(res=np.inf, redHess=np.array([-1.0]), svmin=0, svmax=0, rank_def=-1)
    if not (np.all(np.isfinite(J)) and np.all(np.isfinite(v))):
        return v, lam, bad
    if lam is None:
        try:
            lam = np.linalg.lstsq(J.T, obj_grad, rcond=None)[0]
        except np.linalg.LinAlgError:
            return v, lam, bad
    for it in range(maxit):
        F = S.F(v)
        J = S.J(v)
        gL = obj_grad - J.T @ lam
        res = np.linalg.norm(gL) + np.linalg.norm(F)
        if verbose:
            print('  kkt it', it, 'res', res, '|F|', np.linalg.norm(F))
        if res < tol:
            break
        H = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = h * max(1.0, abs(v[i]))
            H[:, i] = -(S.J(v + e).T @ lam - S.J(v - e).T @ lam) / (2 * e[i])
        H = 0.5 * (H + H.T)
        K = np.block([[H, -J.T], [J, np.zeros((m, m))]])
        rhs = -np.concatenate([gL, F])
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(rhs))):
            return v, lam, dict(res=np.inf, redHess=np.array([-1.0]), svmin=0, svmax=0, rank_def=-1)
        try:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        except np.linalg.LinAlgError:
            return v, lam, dict(res=np.inf, redHess=np.array([-1.0]), svmin=0, svmax=0, rank_def=-1)
        dv = sol[:n]
        sc = np.linalg.norm(v) + 1.0
        if np.linalg.norm(dv) > 0.3 * sc:
            t = 0.3 * sc / np.linalg.norm(dv)
            dv = dv * t
            sol = sol * t
        v = v + dv
        lam = lam + sol[n:]
    # second-order: reduced Hessian
    J = S.J(v)
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(v)):
        return v, lam, dict(res=np.inf, redHess=np.array([-1.0]), svmin=0, svmax=0, rank_def=-1)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h * max(1.0, abs(v[i]))
        H[:, i] = -(S.J(v + e).T @ lam - S.J(v - e).T @ lam) / (2 * e[i])
    H = 0.5 * (H + H.T)
    try:
        Z, s = null_basis(J)
    except np.linalg.LinAlgError:
        return v, lam, dict(res=np.inf, redHess=np.array([-1.0]), svmin=0, svmax=0, rank_def=-1)
    Hr = Z.T @ H @ Z
    ev = np.linalg.eigvalsh(Hr) if Hr.size else np.array([])
    F = S.F(v)
    gL = obj_grad - J.T @ lam
    return v, lam, dict(res=float(np.linalg.norm(gL) + np.linalg.norm(F)), redHess=ev,
                        svmin=float(s[-1] if len(s) else 0), svmax=float(s[0]), rank_def=int(J.shape[0] - np.sum(s > s[0] * 1e-12)))


def slsqp_min(S, v, obj_index=None, obj_fun=None, obj_jac=None, maxiter=2000, ftol=1e-15, bounds=None):
    if obj_index is not None:
        def f(x):
            return x[obj_index]

        def g(x):
            e = np.zeros_like(x)
            e[obj_index] = 1.0
            return e
    else:
        f, g = obj_fun, obj_jac
    cons = [dict(type='eq', fun=S.F, jac=S.J)]
    try:
        r = minimize(f, v, jac=g, constraints=cons, method='SLSQP', bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=ftol))
    except Exception:
        class R: pass
        r = R(); r.x = v; r.message = 'exception'; r.nit = 0
    if not np.all(np.isfinite(r.x)):
        r.x = v
    return r


def diagnostics(z, G, a):
    N = len(z)
    d = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(d, np.inf)
    R = np.max(np.abs(z))
    I = np.sum(G * np.abs(z) ** 2) / np.sum(np.abs(G) * np.abs(z) ** 2)
    iu = np.triu_indices(N, 1)
    if a == 0:
        terms = (G[:, None] * G[None, :])[iu]
    else:
        terms = (G[:, None] * G[None, :] * d ** (-a))[iu]
    H = terms.sum() / np.abs(terms).sum()
    return dict(minpair_rel=float(d.min() / R), Gspan=float(np.abs(G).max() / np.abs(G).min()), I=float(I), H=float(H),
                Gsum=float(G.sum()), Rmax=float(R), Rmin=float(np.abs(z).min()))


def load_config(cfg, a):
    z = np.array([complex(p[0], p[1]) for p in cfg['z']])
    G = np.array(cfg['G'], float)
    return z, G
