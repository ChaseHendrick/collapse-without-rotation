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
"""Theorem 2 of the manuscript (paper/collapse-without-rotation.tex): a strong vortex with one weak triple.

A vortex of circulation 1 at the origin and three vortices of circulations gamma g1, gamma g2, gamma (g3 + gamma nu)
at z_k = 1 + gamma zeta_k, with g1 + g2 + g3 = 0 and zeta1 + zeta2 + zeta3 = 0. The regularized system is
  E_k = 1/(1 + gamma zeta_k) + w_k - Lambda conj(1 + gamma zeta_k - z_c) = 0   (k = 1, 2),
  E_0 = nu - sum_k g_k zeta_k/(1 + gamma zeta_k) - Lambda (nu (1 - conj z_c) + conj D) = 0,
  Im(-2 D/S) = y0,
with z_c = gamma^2 (nu + D)/(1 + gamma^2 nu), D = sum g_k zeta_k and S = sum g_k^2 (g3 carrying gamma nu). At
gamma = 0 it is solved by the equilateral triangle zeta_k = a omega^k, omega = exp(+-2 pi i/3), Lambda = 1 + 1/q0,
nu = S/2, q0 = 1/2 + i y0.

Sections:
  1. EXACT, over the field Q(t)(omega) with g1 = 1, g2 = t (the equations are invariant under
     (gamma, g, zeta) -> (gamma/lam, lam g, lam zeta)): the point solves the system at gamma = 0; at the unit
     triangle zeta_k = omega^k the complex-linear form d(w1 - w2) is g3 (omega^2 dzeta1 - omega dzeta2), and at
     a omega^k it is that form divided by a^2 (so it is nonzero for every y0); and at y0 = sqrt3/2 the derivative of
     P = Re Lambda/(-2 Im Lambda) along the branch is 2 sqrt3 g1 g2 g3/S at gamma = 0, for both omega.
  2. NUMERICAL, 40 digits: the branch by Newton's method at gamma = 1e-2 .. 1e-5 for three triples and both omega;
     every point also satisfies the original Biot-Savart equations of all four vortices; (P - P(0))/gamma tends to
     the exact slope with an error that falls about tenfold per decade of gamma; P < sqrt3/2 when g1 g2 g3 < 0;
     P > sqrt3/2 when g1 g2 g3 > 0; other y0 give P(0) = (y0^2 + 3/4)/(2 y0); the Jacobian at gamma = 0 is
     invertible; for g = (1, 1, -2) the
     configuration lies in the class K(3, 1/2).
  3. Negative controls.

Needs sympy and mpmath (code/requirements.txt). Run: python3 code/verify_triple_branch.py. Prints every check and
exits with status 1 if any fails; its output is data/verify-triple-branch.txt. About ten seconds.
"""
import os
import sys
import sympy as sp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
OUT = []
FAILED = []
N = [0]


def say(s=''):
    print(s)
    OUT.append(s)


def check(name, ok, detail=''):
    N[0] += 1
    say(('OK    ' if ok else 'FAIL  ') + name + (('   [' + str(detail) + ']') if detail else ''))
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- 1. exact, in Q(t)(omega)
t = sp.symbols('t')
cz = sp.cancel


class W:
    """a + b omega with a, b in Q(t), omega^2 + omega + 1 = 0; conj(omega) = omega^2 for real t."""
    __slots__ = ('a', 'b')

    def __init__(s, a, b=0):
        s.a = cz(sp.sympify(a))
        s.b = cz(sp.sympify(b))

    def __add__(s, o):
        o = o if isinstance(o, W) else W(o)
        return W(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __neg__(s):
        return W(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-(o if isinstance(o, W) else W(o)))

    def __rsub__(s, o):
        return W(o) - s

    def __mul__(s, o):
        o = o if isinstance(o, W) else W(o)
        return W(s.a*o.a - s.b*o.b, s.a*o.b + s.b*o.a - s.b*o.b)
    __rmul__ = __mul__

    def conj(s):
        return W(s.a - s.b, -s.b)

    def inv(s):
        n = cz(s.a**2 - s.a*s.b + s.b**2)
        c = s.conj()
        return W(c.a/n, c.b/n)

    def __truediv__(s, o):
        return s*(o if isinstance(o, W) else W(o)).inv()

    def __rtruediv__(s, o):
        return W(o)*s.inv()

    def iszero(s):
        return s.a == 0 and s.b == 0


OM, OM2, HALF = W(0, 1), W(-1, -1), W(sp.Rational(1, 2))


def exact_branch(sigma):
    g = {1: W(1), 2: W(t), 3: W(-1 - t)}
    S = g[1]*g[1] + g[2]*g[2] + g[3]*g[3]
    rot = OM if sigma == 1 else OM2
    pw = lambda k: [W(1), rot, rot*rot][k % 3]
    q0 = 1 + OM                                        # exp(i pi/3): y0 = sqrt3/2
    a = q0*(g[1]/(1 - pw(1)) + g[2]/(1 - pw(2)))       # w_3 of the unit triangle, times q0
    z = {k: a*pw(k) for k in (1, 2, 3)}
    w = lambda k: sum((g[l]/(z[k] - z[l]) for l in (1, 2, 3) if l != k), W(0))
    L0, nu0 = 1 + 1/q0, S*HALF
    D = sum((g[k]*z[k] for k in (1, 2, 3)), W(0))
    name = 'omega = exp(%s2 pi i/3)' % ('+' if sigma == 1 else '-')
    check(name + ': the triangle translates, w_1 = w_2 = w_3 = 1/q0', all((w(k) - 1/q0).iszero() for k in (1, 2, 3)))
    check(name + ': at gamma = 0, E_1 = E_2 = 0, E_0 = 0 and D = -(S/2) q0',
          all((1 + w(k) - L0).iszero() for k in (1, 2)) and (nu0 - D - L0*(nu0 + D.conj())).iszero()
          and (D + S*q0*HALF).iszero())
    dz = lambda l, j: W(1) if l == j else (W(-1) if l == 3 else W(0))

    def dw(k, j):
        return sum((-g[l]*(dz(k, j) - dz(l, j))/((z[k] - z[l])*(z[k] - z[l])) for l in (1, 2, 3) if l != k), W(0))
    ell = [dw(1, j) - dw(2, j) for j in (1, 2)]
    zu = {k: pw(k) for k in (1, 2, 3)}                 # the unit triangle
    def dwu(k, j):
        return sum((-g[l]*(dz(k, j) - dz(l, j))/((zu[k] - zu[l])*(zu[k] - zu[l])) for l in (1, 2, 3) if l != k), W(0))
    ellu = [dwu(1, j) - dwu(2, j) for j in (1, 2)]
    check(name + ': at the unit triangle d(w1 - w2) = g3 (omega^2 dzeta1 - omega dzeta2), nonzero since g3 != 0',
          (ellu[0] - g[3]*rot*rot).iszero() and (ellu[1] + g[3]*rot).iszero())
    check(name + ': at zeta_k = a omega^k the form is the unit-triangle form divided by a^2 (so nonzero for every y0)',
          all((ell[j]*a*a - ellu[j]).iszero() for j in (0, 1)))
    check(name + ': d(w1 - w2) annihilates the scaling direction zeta0, and d w_1 . zeta0 = -1/q0',
          (ell[0]*z[1] + ell[1]*z[2]).iszero() and (dw(1, 1)*z[1] + dw(1, 2)*z[2] + 1/q0).iszero())
    O = W(0)
    rows, rhs = [], []
    for k in (1, 2):
        rows.append([dw(k, 1), dw(k, 2), O, O, W(-1), O, O])
        rhs.append(-(-z[k] + nu0/(z[k] - z[3]) - L0*z[k].conj()))
    dD = {j: g[j] - g[3] for j in (1, 2)}
    rows.append([-dD[1], -dD[2], -L0*dD[1], -L0*dD[2], -(nu0 + D.conj()), O, 1 - L0])
    M = sum((g[k]*z[k]*z[k] for k in (1, 2, 3)), W(0))
    rhs.append(-(M - nu0*z[3] - L0*nu0*z[3].conj()))
    for r, b in list(zip(rows, rhs)):
        rows.append([r[2].conj(), r[3].conj(), r[0].conj(), r[1].conj(), r[5].conj(), r[4].conj(), r[6].conj()])
        rhs.append(b.conj())
    dq = {j: -2*dD[j]/S for j in (1, 2)}
    rows.append([dq[1], dq[2], -dq[1], -dq[2], O, O, O])
    dqg = -2*nu0*z[3]/S + 2*D*(2*g[3]*nu0)/(S*S)
    rhs.append(-(dqg - dqg.conj()))
    A = [r[:] + [b] for r, b in zip(rows, rhs)]
    for c in range(7):
        p = next(r for r in range(c, 7) if not A[r][c].iszero())
        A[c], A[p] = A[p], A[c]
        iv = A[c][c].inv()
        A[c] = [x*iv for x in A[c]]
        for r in range(7):
            if r != c and not A[r][c].iszero():
                f = A[r][c]
                A[r] = [x - f*y for x, y in zip(A[r], A[c])]
    x = [A[r][7] for r in range(7)]
    dL, dLb = x[4], x[5]
    check(name + ': the linear system for the derivative is consistent (d conj(Lambda) = conj(d Lambda))',
          (dLb - dL.conj()).iszero())
    # dP = i B/6 with B = (dL + dLb)(L0 - conj L0) - 3 (dL - dLb); i = (2 omega + 1)/sqrt3 and 2 sqrt3 = 6/sqrt3
    B = (dL + dLb)*(L0 - L0.conj()) - 3*(dL - dLb)
    check(name + ': dP/dgamma at gamma = 0 equals 2 sqrt3 g1 g2 g3/S, identically in t = g2/g1',
          ((2*OM + 1)*B - 36*g[1]*g[2]*g[3]/S).iszero(),
          "Lambda'(0) = [%s] + [%s] omega" % (sp.factor(dL.a), sp.factor(dL.b)))
    check(name + ': d nu/dgamma at gamma = 0 equals g3 S/2', (x[6] - g[3]*S*HALF).iszero(), sp.factor(x[6].a))


say('1. Exact checks over Q(t)(omega), g = (1, t, -1 - t), y0 = sqrt3/2')
for sg in (1, -1):
    exact_branch(sg)

# ---------------------------------------------------------------- 2. numerical, 40 digits
mp.mp.dps = 40
say('\n2. The branch at 40 digits')


def F(x, gam, g1, g2, y0):
    z1, z2, L, nu = mp.mpc(x[0], x[1]), mp.mpc(x[2], x[3]), mp.mpc(x[4], x[5]), x[6]
    zs, gs = [z1, z2, -z1 - z2], [g1, g2, -g1 - g2 + gam*nu]
    D = sum(g*z for g, z in zip(gs, zs))
    S = sum(g*g for g in gs)
    zc = gam**2*(nu + D)/(1 + gam**2*nu)
    out = []
    for k in (0, 1):
        w = sum(gs[l]/(zs[k] - zs[l]) for l in range(3) if l != k)
        e = 1/(1 + gam*zs[k]) + w - L*mp.conj(1 + gam*zs[k] - zc)
        out += [e.real, e.imag]
    e0 = nu - sum(g*z/(1 + gam*z) for g, z in zip(gs, zs)) - L*(nu*(1 - mp.conj(zc)) + mp.conj(D))
    return out + [e0.real, e0.imag, (-2*D/S).imag - y0]


def start(g1, g2, y0, sigma):
    om = mp.exp(sigma*2j*mp.pi/3)
    q0 = mp.mpc(0.5, y0)
    a = q0*(g1/(1 - om) + g2/(1 - om**2))
    z1, z2, L = a*om, a*om**2, 1 + 1/q0
    return [z1.real, z1.imag, z2.real, z2.imag, L.real, L.imag, ((-g1 - g2)**2 + g1**2 + g2**2)/2]


def jac(x, gam, *p):
    h = mp.mpf(10)**-18
    cols = []
    for i in range(7):
        xp, xm = list(x), list(x)
        xp[i] += h
        xm[i] -= h
        cols.append([(u - v)/(2*h) for u, v in zip(F(xp, gam, *p), F(xm, gam, *p))])
    return mp.matrix([[cols[c][r] for c in range(7)] for r in range(7)])


def newton(x, gam, *p):
    x = list(x)
    for _ in range(60):
        dx = mp.lu_solve(jac(x, gam, *p), -mp.matrix(F(x, gam, *p)))
        x = [u + v for u, v in zip(x, dx)]
        if mp.norm(dx) < mp.mpf(10)**-34:
            break
    return x


def config(x, gam, g1, g2):
    z1, z2 = mp.mpc(x[0], x[1]), mp.mpc(x[2], x[3])
    zs = [mp.mpc(0), 1 + gam*z1, 1 + gam*z2, 1 + gam*(-z1 - z2)]
    G = [mp.mpf(1), gam*g1, gam*g2, gam*(-g1 - g2 + gam*x[6])]
    return zs, G, mp.mpc(x[4], x[5])


def biot_savart_residual(zs, G, L):
    zc = sum(g*z for g, z in zip(G, zs))/sum(G)
    worst = 0
    for j in range(4):
        lhs = sum(G[k]/(zs[j] - zs[k]) for k in range(4) if k != j)
        rhs = L*mp.conj(zs[j] - zc)
        worst = max(worst, abs(lhs - rhs)/max(abs(lhs), 1))
    return worst


def Pof(L):
    return abs(L.real)/(-2*L.imag)


y0 = mp.sqrt(3)/2
triples = [(mp.mpf(1), mp.mpf(1)), (mp.mpf(1), mp.mpf('0.6')), (mp.mpf('-1.3'), mp.mpf('0.5')), (mp.mpf(-1), mp.mpf('-0.6'))]
for g1, g2 in triples:
    g3 = -g1 - g2
    S = g1**2 + g2**2 + g3**2
    slope = 2*mp.sqrt(3)*g1*g2*g3/S
    for sg in (1, -1):
        tag = 'g = (%s, %s, %s), omega = exp(%s2 pi i/3)' % (mp.nstr(g1, 3), mp.nstr(g2, 3), mp.nstr(g3, 3),
                                                         '+' if sg == 1 else '-')
        x0 = start(g1, g2, y0, sg)
        check(tag + ': gamma = 0 residual below 1e-35 and det of the Jacobian nonzero',
              mp.norm(mp.matrix(F(x0, 0, g1, g2, y0))) < mp.mpf(10)**-35 and abs(mp.det(jac(x0, 0, g1, g2, y0))) > 1e-3,
              mp.nstr(mp.det(jac(x0, 0, g1, g2, y0)), 8))
        x, errs, worst, imL, Ps = x0, [], 0, [], []
        for e in range(2, 6):
            gam = mp.mpf(10)**-e
            x = newton(x, gam, g1, g2, y0)
            zs, G, L = config(x, gam, g1, g2)
            worst = max(worst, biot_savart_residual(zs, G, L))
            imL.append(L.imag)
            Ps.append(Pof(L))
            errs.append(abs((Pof(L) - y0)/gam - slope))
        check(tag + ': every branch point satisfies the Biot-Savart equations of all four vortices to 1e-30',
              worst < mp.mpf(10)**-30, mp.nstr(worst, 3))
        check(tag + ': Im Lambda < 0 (a collapse) at every gamma', all(v < 0 for v in imL))
        ratios = [errs[i]/errs[i + 1] for i in range(len(errs) - 1)]
        check(tag + ': (P - sqrt3/2)/gamma -> 2 sqrt3 g1g2g3/S = %s, error falling about tenfold per decade'
              % mp.nstr(slope, 10), all(8 < r < 12 for r in ratios) and errs[-1] < 1e-3,
              'errors ' + ', '.join(mp.nstr(v, 3) for v in errs))
        if g1*g2*g3 < 0:
            check(tag + ': g1 g2 g3 < 0, so P < sqrt3/2 at gamma = 1e-2 .. 1e-5', all(p < y0 for p in Ps),
                  'P(1e-2) = ' + mp.nstr(Ps[0], 12))
        else:
            check(tag + ': g1 g2 g3 > 0, so P > sqrt3/2 at gamma = 1e-2 .. 1e-5 (same y0)', all(p > y0 for p in Ps))

for yy in (mp.mpf('0.4'), mp.mpf('1.5')):
    g1, g2 = mp.mpf(1), mp.mpf('0.6')
    x = newton(start(g1, g2, yy, 1), mp.mpf(10)**-6, g1, g2, yy)
    L = mp.mpc(x[4], x[5])
    target = (yy**2 + mp.mpf(3)/4)/(2*yy)
    check('y0 = %s: det of the Jacobian at gamma = 0 nonzero, and P(1e-6) - (y0^2 + 3/4)/(2 y0) = O(gamma)'
          % mp.nstr(yy, 2), abs(mp.det(jac(start(g1, g2, yy, 1), 0, g1, g2, yy))) > 1e-3
          and abs(Pof(L) - target) < 1e-4, mp.nstr(Pof(L) - target, 3))

# The class K(3, c) with c = 1/2 for g = (1, 1, -2): c <= |g_k| <= 1/c, |zeta_k| <= 1/c, |zeta_k - zeta_l| >= c.
g1, g2 = mp.mpf(1), mp.mpf(1)
x = start(g1, g2, y0, 1)
ok = True
for e in range(2, 6):
    gam = mp.mpf(10)**-e
    x = newton(x, gam, g1, g2, y0)
    z1, z2 = mp.mpc(x[0], x[1]), mp.mpc(x[2], x[3])
    zs = [z1, z2, -z1 - z2]
    gs = [g1, g2, -g1 - g2 + gam*x[6]]
    ok &= all(mp.mpf(1)/2 <= abs(v) <= 2 for v in gs) and all(abs(v) <= 2 for v in zs)
    ok &= all(abs(zs[i] - zs[j]) >= mp.mpf(1)/2 for i in range(3) for j in range(i))
check('g = (1, 1, -2): the branch lies in K(3, 1/2) (Z_C = 1) at gamma = 1e-2 .. 1e-5; there P < sqrt3/2', ok)

# ---------------------------------------------------------------- 3. controls
say('\n3. Negative controls (must fail)')
x0 = start(mp.mpf(1), mp.mpf('0.6'), y0, 1)
bad = list(x0)
bad[6] = bad[6]*mp.mpf('1.01')
check('control: nu = 1.01 S/2 does not solve the system at gamma = 0',
      mp.norm(mp.matrix(F(bad, 0, mp.mpf(1), mp.mpf('0.6'), y0))) > 1e-3)
def undivided(x, gam, g1, g2):
    z1, z2, L, nu = mp.mpc(x[0], x[1]), mp.mpc(x[2], x[3]), mp.mpc(x[4], x[5]), x[6]
    zs, gs = [z1, z2, -z1 - z2], [g1, g2, -g1 - g2 + gam*nu]
    D = sum(g*z for g, z in zip(gs, zs))
    zc = gam**2*(nu + D)/(1 + gam**2*nu)
    return sum(g*(1/(1 + gam*z) + sum(gs[l]/(z - zs[l]) for l in range(3) if zs[l] != z) - L*mp.conj(1 + gam*z - zc))
               for g, z in zip(gs, zs))
check('control: without the factor 1/gamma, sum g_k E_k vanishes at gamma = 0 also for nu = 1.01 S/2, so it cannot fix nu',
      abs(undivided(bad, 0, mp.mpf(1), mp.mpf('0.6'))) < mp.mpf(10)**-35, mp.nstr(abs(undivided(bad, 0, mp.mpf(1), mp.mpf('0.6'))), 3))
wrong = 2*mp.sqrt(3)*mp.mpf(1)*mp.mpf('0.6')*mp.mpf('-1.6')/mp.mpf('3.92')*2
xg = newton(x0, mp.mpf(10)**-5, mp.mpf(1), mp.mpf('0.6'), y0)
check('control: twice the slope is not the observed slope at gamma = 1e-5',
      abs((Pof(mp.mpc(xg[4], xg[5])) - y0)/mp.mpf(10)**-5 - wrong) > 0.5)

say('\n%d checks, %d failed' % (N[0], len(FAILED)))
with open(os.path.join(DATA, 'verify-triple-branch.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
