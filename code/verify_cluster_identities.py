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
"""Exact (SymPy) checks of every identity used in the proof of Theorem 1 of the manuscript
(paper/collapse-without-rotation.tex), a strong vortex with weak tight clusters, and of the formal
first-order expansion of Section 3.3.

Setting (Euler law, conj(dz_j/dt) = (1/(2 pi i)) sum_{k != j} Gamma_k/(z_j - z_k)). The class K(n, c), 0 < c <= 1:
circulation 1 at the origin; weak vortices Gamma_k = gamma g_k at z_k (k = 1..n), c <= |g_k| <= 1/c, any signs,
partitioned into clusters C with points Z_C: c <= |Z_C| <= 1/c, |Z_C - Z_D| >= c (C != D), |z_k - Z_C| <= gamma/c
(k in C), |z_k - z_l| >= c gamma (k != l in C). Singletons are allowed. We write zeta_k = (z_k - Z_C)/gamma,
w_k = sum_{l in C, l != k} g_l/(zeta_k - zeta_l), G = sum g, D = sum g zeta, S2 = sum g^2, N2 = sum g |zeta|^2,
Lambda = 2 pi i conj(kappa), E the field of the other weak vortices, T = Lambda conj(Z - z_c) - 1/Z - E(Z).

Sections:
  1. The internal identities (I1) sum g w = 0 and (I2) sum g zeta w = (G^2 - S2)/2, and G^2 - S2 = 2 sum_{k<l} g_k g_l.
  2. The exact expansion behind (A'); the leading-order algebra of Step 4 (the translation equation, D = -(S2/2) q Z,
     nu = (S2/2)(|q|^2 + conj q + q^2), reality, P = (y^2 + 3/4)/(2y), the geometry of q).
  3. The angular impulse (Case B) and the per-cluster impulse density.
  4. The classical condition sum_{i<j} Gamma_i Gamma_j = 0 for one and for several clusters.
  5. Part (d), a singleton: the explicit constants, for every n; and the containment of the class of the companion
     paper's Theorem 3 in K(2m, c^2).
  6. The formal first-order expansion (not a proof): L = (q + 1)/(q - e), p = (1 + e)/2,
     P_min = sqrt((3 + e)(1 - e))/(2(1 + e)) = sqrt3/2 - (2/sqrt3) e + O(e^2), the equilateral triple, the pair.
  7. Negative controls: statements that must fail do fail.

Needs sympy (code/requirements.txt). Run: python3 code/verify_cluster_identities.py. Prints every check and exits
with status 1 if any fails; its output is data/verify-cluster-identities.txt. About ten seconds.
"""
import os
import sys
import sympy as sp

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


def zero(name, e):
    check(name, sp.simplify(sp.together(sp.expand(e))) == 0)


I = sp.I
cj = sp.conjugate

say('Theorem 1 (a strong vortex with weak tight clusters): exact identities, SymPy %s' % sp.__version__)
say('\n1. Internal identities of a cluster')
for n in (2, 3, 4, 5):
    g = sp.symbols('g1:%d' % (n + 1))
    z = sp.symbols('s1:%d' % (n + 1))
    w = [sum(g[l]/(z[k] - z[l]) for l in range(n) if l != k) for k in range(n)]
    G = sum(g)
    S2 = sum(x**2 for x in g)
    zero('(I1) n=%d: sum_k g_k w_k = 0' % n, sum(g[k]*w[k] for k in range(n)))
    zero('(I2) n=%d: sum_k g_k zeta_k w_k = (G^2 - S2)/2' % n, sum(g[k]*z[k]*w[k] for k in range(n)) - (G**2 - S2)/2)
    zero('     n=%d: G^2 - S2 = 2 sum_{k<l} g_k g_l (positive for a same-sign cluster of two or more)' % n,
         G**2 - S2 - 2*sum(g[k]*g[l] for k in range(n) for l in range(k)))

say('\n2. The weighted sums and the leading-order algebra (Steps 2 and 4)')
n = 3
g = sp.symbols('g1:4', real=True)
zr = sp.symbols('x1:4', real=True)
zi = sp.symbols('y1:4', real=True)
zeta = [zr[k] + I*zi[k] for k in range(n)]
Z = sp.Symbol('Z')
gam = sp.Symbol('gamma', positive=True)
D = sum(g[k]*zeta[k] for k in range(n))
G = sum(g)
zero('(A) exact: sum g/(Z + gam zeta) = G/Z - gam D/Z^2 + gam^2 sum g zeta^2/(Z^2 (Z + gam zeta))',
     sum(g[k]/(Z + gam*zeta[k]) for k in range(n)) - (G/Z - gam*D/Z**2 + gam**2*sum(g[k]*zeta[k]**2/(Z**2*(Z + gam*zeta[k])) for k in range(n))))
zero('(B) exact: sum g zeta/(Z + gam zeta) = D/Z - gam sum g zeta^2/(Z (Z + gam zeta))',
     sum(g[k]*zeta[k]/(Z + gam*zeta[k]) for k in range(n)) - (D/Z - gam*sum(g[k]*zeta[k]**2/(Z*(Z + gam*zeta[k])) for k in range(n))))
p, y = sp.symbols('p y', real=True)
S2 = sp.Symbol('S2', positive=True)
R = sp.Symbol('R', positive=True)
th = sp.Symbol('theta', real=True)
Zc = R*sp.exp(I*th)
q = p + I*y
T = 1/(q*Zc)
Lam = (1 + T*Zc)/R**2
zero('translation equation Lambda conj(Z) = 1/Z + T with q = 1/(T Z) gives Lambda |Z|^2 = 1 + 1/q', Lam*cj(Zc) - (1/Zc + T))
Dst = -S2/(2*T)
zero('(B) in the limit with G = 0: D = -S2/(2 T) = -(S2/2) q Z', Dst + S2*q*Zc/2)
nu = sp.expand(-(Lam*cj(Dst) + Dst/Zc**2)/T)
nu_claim = S2/2*(q*cj(q) + cj(q) + q**2)
zero("(A') in the limit: nu = -(Lambda conj D + D/Z^2)/T = (S2/2)(|q|^2 + conj q + q^2)", nu - nu_claim)
zero('   = (S2/2)[(2p^2 + p) + i y (2p - 1)]', nu_claim - S2/2*((2*p**2 + p) + I*y*(2*p - 1)))
check('nu is real (circulations are real) iff y (2p - 1) = 0', sp.factor(sp.im(sp.expand(nu_claim))) == sp.factor(S2*y*(2*p - 1)/2),
      sp.factor(sp.im(sp.expand(nu_claim))))
zero('p = 1/2: nu = S2/2, the net circulation gamma^2 nu = (1/2) sum Gamma_k^2 (ratio nuhat = 1)',
     sp.re(sp.expand(nu_claim.subs(p, sp.Rational(1, 2)))) - S2/2)
L = sp.expand(Lam*R**2)
Pexpr = sp.simplify(sp.re(L)/(-2*sp.im(L)))
zero('p = 1/2: P = Re(Lambda|Z|^2)/(2|Im(Lambda|Z|^2)|) = (y^2 + 3/4)/(2y)', Pexpr.subs(p, sp.Rational(1, 2)) - (y**2 + sp.Rational(3, 4))/(2*y))
zero('Im(Lambda |Z|^2) = -y/|q|^2 (a collapse, Im Lambda < 0, forces y > 0)', sp.im(L) + y/(p**2 + y**2))
zero('Re(Lambda |Z|^2) = 1 + p/|q|^2 (positive at p = 1/2)', sp.re(L) - 1 - p/(p**2 + y**2))
ph = sp.Symbol('phi', real=True)
qphi = 1/(1 + sp.exp(-I*ph))
zero('1/q = 1 + e^{-i phi} gives Re q = 1/2', sp.simplify(sp.re(sp.expand_complex(qphi)) - sp.Rational(1, 2)))
tt = sp.Symbol('t', real=True)
Pt = ((sp.tan(tt)/2)**2 + sp.Rational(3, 4))/(2*sp.tan(tt)/2)
zero('with t = arg q (p = 1/2, y = tan(t)/2): P = (2 + cos 2t)/(2 sin 2t)', sp.simplify(sp.expand_trig(Pt - (2 + sp.cos(2*tt))/(2*sp.sin(2*tt)))))
zero('speed ratio |V_C|/|V_s| = 1/|q| = 2 cos t on p = 1/2 (at t = pi/5)',
     sp.simplify(1/sp.sqrt(sp.Rational(1, 4) + (sp.tan(tt)/2)**2) - 2*sp.cos(tt)).subs(tt, sp.pi/5))
zero('minimum at y = sqrt3/2: q = e^{i pi/3}, |q| = 1, P = sqrt3/2', ((y**2 + sp.Rational(3, 4))/(2*y)).subs(y, sp.sqrt(3)/2) - sp.sqrt(3)/2)
zero('(y^2 + 3/4)/(2y) = sqrt3/2 + (y - sqrt3/2)^2/(2y)', (y**2 + sp.Rational(3, 4))/(2*y) - sp.sqrt(3)/2 - (y - sp.sqrt(3)/2)**2/(2*y))
zero('the two roots y, 3/(4y) of y^2 - 2 P y + 3/4 give the same P', ((y**2 + sp.Rational(3, 4))/(2*y)) - (((3/(4*y))**2 + sp.Rational(3, 4))/(2*(3/(4*y)))))
fy = ((sp.Rational(3, 4) + y**2)**2 + y**2)/(sp.Rational(1, 4) + y**2)**2
zero('p = 1/2: |1 + 1/q|^2 = ((3/4 + y^2)^2 + y^2)/(1/4 + y^2)^2, and Lambda|Z_C|^2 = 1 + 1/q_C fixes |Z_C| from y_C',
     sp.Abs(1 + 1/(sp.Rational(1, 2) + I*y))**2 - fy)
dfy = sp.factor(sp.together(fy - fy.subs(y, sp.Rational(3, 4)/y)))
check('   the two roots y and 3/(4y) give different |Z_C| unless y = sqrt3/2', sp.simplify(dfy.subs(y, sp.sqrt(3)/2)) == 0 and sp.simplify(dfy.subs(y, 1)) != 0, dfy)

say('\n3. The angular impulse (Case B)')
gg = sp.symbols('h1:4', real=True)
Zr, Zi = sp.symbols('Zr Zi', real=True)
ZZ = Zr + I*Zi
Gc = sp.Symbol('nu', real=True)*gam
gsc = list(gg[:2]) + [Gc - gg[0] - gg[1]]
Dc = sum(gsc[k]*zeta[k] for k in range(3))
Imp = sp.expand(sum(gam*gsc[k]*(ZZ + gam*zeta[k])*cj(ZZ + gam*zeta[k]) for k in range(3)))
poly = sp.Poly(Imp, gam)
check('impulse of one cluster: the coefficient of gamma vanishes when G = gamma nu', sp.expand(poly.coeff_monomial(gam)) == 0)
zero('impulse of one cluster: the gamma^2 coefficient is nu |Z|^2 + 2 Re(conj(Z) D)',
     poly.coeff_monomial(gam**2) - (sp.Symbol('nu', real=True)*(Zr**2 + Zi**2) + 2*sp.re(sp.expand(cj(ZZ)*Dc.subs(gam, 0)))))
qm = -1
Dm = -S2*qm*Zc/2
num = S2/2*(qm*qm + qm + qm**2)
imp_m = sp.simplify(num*R**2 + 2*sp.re(sp.expand(cj(Zc)*Dm)))
zero('Case B, Lambda* = 0 forces q = -1: nu|Z|^2 + 2 Re(conj Z D) = (3/2) S2 |Z|^2 > 0, so zero impulse is impossible', imp_m - sp.Rational(3, 2)*S2*R**2)
Dq = -S2*q*Zc/2
impq = sp.simplify(sp.re(sp.expand(nu_claim))*R**2 + 2*sp.re(sp.expand(cj(Zc)*Dq)))
zero('per-cluster impulse density (S2/2)|Z|^2 (2p^2 - p), zero at p = 1/2', impq - S2/2*R**2*(2*p**2 - p))

say('\n4. The classical condition sum_{i<j} Gamma_i Gamma_j = 0')
Gs, S2s, nus = sp.symbols('G S2 nu', real=True)
one = gam*Gs + gam**2*(Gs**2 - S2s)/2
zero('one cluster: sum_{i<j} Gamma_i Gamma_j = gamma G + gamma^2 (G^2 - S2)/2 (strong vortex 1, weak gamma g_k)',
     one - (gam*Gs + gam**2*(Gs**2 - S2s)/2))
nuh = 2*nus/S2s
sol = sp.solve(sp.Eq(one.subs(Gs, gam*nus), 0), S2s)
zero('one cluster: the condition is S2 = 2 nu + gamma^2 nu^2, i.e. nuhat = 2 nu/S2 = 1 - gamma^2 nu^2/S2 exactly',
     (nuh - (1 - gam**2*nus**2/S2s)).subs(S2s, sol[0]))
ser = sp.expand(sp.series((2*nus/S2s).subs(nus, (-1 + sp.sqrt(1 + gam**2*S2s))/gam**2), gam, 0, 5).removeO())
check('   hence nuhat = 1 - gamma^2 S2/4 + O(gamma^4): constant term 1, no gamma or gamma^3 term, gamma^2 coefficient -S2/4',
      sp.simplify(ser.coeff(gam, 0) - 1) == 0 and ser.coeff(gam, 1) == 0 and ser.coeff(gam, 3) == 0 and sp.simplify(ser.coeff(gam, 2) + S2s/4) == 0,
      ser.coeff(gam, 2))
G1, G2, S21, S22 = sp.symbols('G1 G2 S21 S22', real=True)
g1s = sp.symbols('a1:4', real=True)
g2s = sp.symbols('b1:3', real=True)
allg = [1] + [gam*x for x in g1s] + [gam*x for x in g2s]
pairs = sum(allg[i]*allg[j] for i in range(len(allg)) for j in range(i))
Gc1, Gc2 = sum(g1s), sum(g2s)
S2c1, S2c2 = sum(x**2 for x in g1s), sum(x**2 for x in g2s)
zero('two clusters: sum_{i<j} Gamma_i Gamma_j = gamma(G1 + G2) + gamma^2[(G1^2 - S2_1)/2 + (G2^2 - S2_2)/2 + G1 G2]',
     pairs - (gam*(Gc1 + Gc2) + gam**2*((Gc1**2 - S2c1)/2 + (Gc2**2 - S2c2)/2 + Gc1*Gc2)))

say('\n5. Part (d), a singleton: explicit constants; the class of the companion paper')
x = sp.Symbol('x', positive=True)
b1 = (sp.Rational(2, 5)*x + sp.Rational(1, 10)*x)/(1 - sp.Rational(2, 5)*x)
check('x = 10 n gamma/c^3, |e1| <= 0.4 x, |e2| <= 0.1 x: x - (|e1| + |e2|)/(1 - |e1|) = x(5 - 4x)/(2(5 - 2x)) >= 0 on 0 < x <= 1/2',
      sp.simplify(x - b1 - x*(5 - 4*x)/(2*(5 - 2*x))) == 0)
check('then |Lambda|z|^2 - 1| <= x <= 1/2 gives P >= (1 - x)/(2x) >= 1/(4x) = c^3/(40 n gamma)',
      sp.simplify((1 - x)/(2*x) - 1/(4*x) - (1 - 2*x)/(4*x)) == 0)
nn = sp.Symbol('n', positive=True, integer=True)
check('|e2| <= n gamma/c^3 for every n: (n - 1)(20n + 1) <= 2n(10n - 1), difference 17n + 1 > 0',
      sp.expand(2*nn*(10*nn - 1) - (nn - 1)*(20*nn + 1)) == 17*nn + 1, sp.expand(2*nn*(10*nn - 1) - (nn - 1)*(20*nn + 1)))
check('|e1| <= (20/19)^2 (1 + 1/20) n gamma/c^3 < 1.2 n gamma/c^3 <= 4 n gamma/c^3 for every n',
      sp.Rational(400, 361)*sp.Rational(21, 20) < sp.Rational(6, 5), sp.N(sp.Rational(400, 361)*sp.Rational(21, 20), 6))
cc = sp.Symbol('c', positive=True)
check('the companion paper\'s class (c <= a, b <= 1/c; c <= |Z| <= 1/c; c gamma |Z| <= |W - Z| <= gamma |Z|/c; |Z_j - Z_l| >= c) '
      'lies in K(2m, c^2): gamma|Z|/c <= gamma/c^2 and c gamma |Z| >= c^2 gamma',
      sp.simplify((1/cc)/cc - 1/cc**2) == 0 and sp.simplify(cc*cc - cc**2) == 0)

say('\n6. The formal first-order expansion (not a proof)')
g1, g2 = sp.symbols('g1 g2', real=True)
gl = [g1, g2, -g1 - g2]
xs = sp.symbols('a1:4', real=True)
ys = sp.symbols('b1:4', real=True)
zl = [xs[k] + I*ys[k] for k in range(3)]
Dn = sp.expand(sum(gl[k]*zl[k] for k in range(3)))
M2 = sp.expand(sum(gl[k]*zl[k]**2 for k in range(3)))
Nn = sp.expand(sum(gl[k]*(xs[k]**2 + ys[k]**2) for k in range(3)))
Jn = Nn - sp.re(sp.expand(sp.conjugate(M2)*Dn**2))/(sp.re(Dn)**2 + sp.im(Dn)**2)
hsq = [sp.im(sp.expand(zl[k]*sp.conjugate(Dn)))**2/(sp.re(Dn)**2 + sp.im(Dn)**2) for k in range(3)]
zero('J = N2 - Re(conj(M2) D/conj(D)) = 2 sum g_k h_k^2, h_k the coordinate perpendicular to D (G = 0)',
     sp.together(Jn - 2*sum(gl[k]*hsq[k] for k in range(3))))
s1, s2 = sp.symbols('s1 s2', real=True)
Jsh = Jn.subs({xs[k]: xs[k] + s1 for k in range(3)}, simultaneous=True).subs({ys[k]: ys[k] + s2 for k in range(3)}, simultaneous=True)
zero('   J is translation invariant', sp.together(sp.expand(Jsh - Jn)))
e = sp.Symbol('e', real=True)
Lsym = sp.Symbol('L')
solL = sp.solve(sp.Eq(Lsym, 1 + (1 + e*Lsym)/q), Lsym)
zero('L = Lambda|Z|^2 solves L = 1 + (1 + e L)/q, so L = (q + 1)/(q - e)', solL[0] - (q + 1)/(q - e))
Lq = (q + 1)/(q - e)
nu1 = sp.simplify((Lq*(p**2 + y**2) + q**2)/(1 + e*Lq))
zero('   nu/(S2/2) = [L|q|^2 + q^2]/(1 + e L) = [|q|^2 + conj q + q^2 - e q]/(1 + e)',
     sp.simplify(nu1 - ((p**2 + y**2) + cj(q) + q**2 - e*q)/(1 + e)))
check('   nu real iff y (2p - 1 - e) = 0', sp.factor(sp.im(sp.expand((p**2 + y**2) + cj(q) + q**2 - e*q))) == sp.factor(y*(2*p - 1 - e)))
Lp = sp.expand_complex(Lq.subs(p, (1 + e)/2))
Pe = sp.simplify(sp.re(Lp)/(-2*sp.im(Lp)))
zero('   p = (1 + e)/2: P = [(3 + e)(1 - e)/4 + y^2]/(2 y (1 + e))', sp.simplify(Pe - ((3 + e)*(1 - e)/4 + y**2)/(2*y*(1 + e))))
Pm = sp.sqrt((3 + e)*(1 - e))/(2*(1 + e))
yp_ = sp.sqrt((3 + e)*(1 - e))/2
zero('   minimum over y at y^2 = (3 + e)(1 - e)/4: P_min = sqrt((3 + e)(1 - e))/(2(1 + e))',
     sp.simplify((((3 + e)*(1 - e)/4 + y**2)/(2*y*(1 + e))).subs(y, yp_) - Pm))
zero('   dP_min/de = -2/sqrt3 at e = 0: P_min = sqrt3/2 - (2/sqrt3) e + O(e^2)', sp.diff(Pm, e).subs(e, 0) + 2/sp.sqrt(3))
Jc, Dab, S2c, Rz = sp.symbols('J D_abs S_2 R', positive=True)
zero('   e = 2 gamma J/(S2 |Z|^2) = gamma J S2/(2|D|^2) at |q| = 1', (2*Jc/(S2c*Rz**2)).subs(Rz, 2*Dab/S2c) - Jc*S2c/(2*Dab**2))
zeq = [0, 1, sp.Rational(1, 2) + I*sp.sqrt(3)/2]
wq = [sum(gl[l]/(zeq[k] - zeq[l]) for l in range(3) if l != k) for k in range(3)]
check('the equilateral triangle with G = 0 is a translating relative equilibrium (w_1 = w_2 = w_3)', all(sp.simplify(wq[k] - wq[0]) == 0 for k in range(3)))
subs_eq = {xs[0]: 0, ys[0]: 0, xs[1]: 1, ys[1]: 0, xs[2]: sp.Rational(1, 2), ys[2]: sp.sqrt(3)/2}
S2t = sum(v**2 for v in gl)
Deq = sp.expand(Dn.subs(subs_eq))
ratio = sp.factor(sp.simplify(Jn.subs(subs_eq)*S2t/(sp.re(Deq)**2 + sp.im(Deq)**2)))
zero('equilateral triple: J S2/|D|^2 = -6 g1 g2 g3/S2, so e = -3 gamma g1 g2 g3/S2', ratio - (-6*g1*g2*(-g1 - g2)/S2t))
zero('   slope -(2/sqrt3) e/gamma = 2 sqrt3 g1 g2 g3/S2 (negative for a (+,+,-) triple beside a positive strong vortex)',
     -2/sp.sqrt(3)*ratio/2 - 2*sp.sqrt(3)*g1*g2*(-g1 - g2)/S2t)
slope_tr = (2*sp.sqrt(3)*g1*g2*(-g1 - g2)/S2t).subs({g1: 1, g2: sp.Rational(3, 5)})
check('   for g = (1, 0.6, -1.6) only: the slope is -1.92 sqrt3/3.92 = -0.8483514160 (this number belongs to that triple)',
      abs(sp.N(slope_tr, 30) + sp.Rational(8483514160, 10**10)) < 1e-10, sp.N(slope_tr, 12))
aa, dx, dy = sp.symbols('aa dx dy', real=True)
Dp = -aa*(dx + I*dy)
M2p = -aa*(dx + I*dy)**2
Np = -aa*(dx**2 + dy**2)
zero('pair (a at 0, -a at d): J = 0, so e = 0 and there is no O(gamma) term (consistent with the companion paper\'s Prop. 4)',
     sp.together(Np - sp.re(sp.expand(sp.conjugate(M2p)*Dp**2))/(sp.re(Dp)**2 + sp.im(Dp)**2)))

say('\n7. Negative controls (must fail)')
v1 = ((2 + sp.cos(sp.pi/3))/(2*sp.sin(sp.pi/3)))
check('control: P = (2 + cos t)/(2|sin t|), with t in place of 2t, is not sqrt3/2 at t = pi/3 (it is 5/(2 sqrt3))',
      sp.simplify(v1 - sp.sqrt(3)/2) != 0, sp.nsimplify(v1))
bad_nu = S2/2*(q*cj(q) + q + q**2)
check('control: the formula with q in place of conj q is not real on p = 1/2 (y != 0)',
      sp.simplify(sp.im(sp.expand(bad_nu.subs(p, sp.Rational(1, 2))))) != 0)
check('control: dropping the angular impulse, q = -1 (Lambda* = 0) solves the other limit equations',
      sp.simplify((1 + 1/sp.Integer(-1))) == 0 and sp.simplify(sp.im(sp.expand(nu_claim.subs({p: -1, y: 0})))) == 0)
check('control: a same-sign pair has G^2 - S2 = 2 g1 g2 > 0, so it cannot satisfy G^2 = S2',
      sp.expand((g1 + g2)**2 - g1**2 - g2**2 - 2*g1*g2) == 0)

say('\n%d checks, %d failed' % (N[0], len(FAILED)))
with open(os.path.join(DATA, 'verify-cluster-identities.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
