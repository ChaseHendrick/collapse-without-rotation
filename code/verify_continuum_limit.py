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
"""The continuum model of Section 6 of paper/collapse-without-rotation.tex: two point vortices and a positive
vortex sheet in self-similar collapse, and its least winding P_inf. NUMERICAL (high precision, not a proof).

  1. The stored solutions data/continuum/continuum-m<m>-n<n>.json (m Chebyshev terms, n quadrature nodes; refined by
     continuum_arb.py at 192 or 256 bits; the collocation system is overdetermined and solved in the least-squares
     sense, so its residual measures the truncation error): the residual falls with m, and P from the 256-bit
     solutions converges, the m = 120 and m = 140 values differing by 1.3e-40, which fixes 40 digits of P_inf.
  2. The two finest recomputed in Arb at 256 bits (continuum_arb.HP): residual, sheet circulation 2 + sqrt2 and zero
     angular impulse.
  3. Exact (SymPy): the continuum form of the classical condition sum_{i<j} Gamma_i Gamma_j = 0 for two vortices -1 and
     a sheet of circulation S, 1 - 2S + S^2/2 = 0, whose roots are 2 +- sqrt2.
  4. Independent: the residual of the continuous equations (not the collocation system) at points between the
     collocation points, with the principal value by singularity subtraction and mpmath quadrature at 40 digits,
     from the stored Chebyshev coefficients (rounded to binary64, so about 1e-14 is the expected floor).
  5. Local minimality along the family: in binary64 at m = 40, the discrete solution set is one-dimensional near the
     solution (one small singular value), dP/dc = 0 and d^2P/dc^2 > 0 along it.
  6. Agreement with the finite two-arm family of the companion paper (data/twoarm-family.json): a fit in powers of
     1/N over N >= 301 and a Richardson extrapolation, against P_inf.
  7. Shape: tip, vortex position, Lambda, arc length, positive density, arm circulation 1 + 1/sqrt2.
  8. No closed form: integer-relation (PSLQ) and algebraic searches with 40 digits, restricted to searches whose
     noise level (the size of the spurious relations that 40 digits allow) exceeds the coefficient bound 1e6.

Needs numpy, mpmath, sympy, python-flint. Run: python3 code/verify_continuum_limit.py. Prints every check and exits
with status 1 if any fails; output in data/verify-continuum-limit.txt. About a minute and a half.
"""
import os
import sys
import glob
import json
import time
import numpy as np
import mpmath as mp
import sympy as sp
from numpy.polynomial import chebyshev as C
from flint import arb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from continuum_model import Model, gn, nullvec, solve_plane  # noqa: E402
from continuum_arb import HP, P_of  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), 'data')
CD = os.path.join(DATA, 'continuum')
OUT = []
FAILED = []
NCHK = [0]
T0 = time.time()
P40 = '0.4773635336916120248430486309060121246263'


def say(s=''):
    print(s)
    sys.stdout.flush()
    OUT.append(s)


def check(name, ok, detail=''):
    NCHK[0] += 1
    say(('OK    ' if ok else 'FAIL  ') + name + (('   [' + detail + ']') if detail else ''))
    if not ok:
        FAILED.append(name)


say('The continuum limit: two point vortices and a positive vortex sheet (NUMERICAL)')
say('\n1. The stored solutions at eight resolutions')
mp.mp.dps = 60
sols = {}
for f in sorted(glob.glob(os.path.join(CD, 'continuum-m*-n*.json'))):
    d = json.load(open(f))
    sols[(d['m'], d['n'])] = d
def P_from_u(d):
    """P from the stored unknowns (-Re Lambda/(2 Im Lambda)); the 192-bit runs stored about 31 digits, the 256-bit runs 55."""
    m = d['m']
    return -mp.mpf(d['u'][2*m + 3])/(2*mp.mpf(d['u'][2*m + 4]))


ref = P_from_u(sols[(140, 840)])
for key in sorted(sols):
    d = sols[key]
    say('   m = %3d, n = %3d, %d bits (stored to %d digits): P = %s  collocation residual %.1e  |P - P(140, 840)| = %s' %
        (key[0], key[1], d['prec'], len(d['u'][0]) - 2, mp.nstr(P_from_u(d), 45), d['resid'], mp.nstr(abs(P_from_u(d) - ref), 3)))
ms = sorted(set(k[0] for k in sols))
best_res = {m: min(sols[k]['resid'] for k in sols if k[0] == m) for m in ms}
check('the collocation residual falls with the number m of Chebyshev terms: %s' % ', '.join('m = %d: %.1e' % (m, best_res[m]) for m in ms),
      all(best_res[ms[i + 1]] < best_res[ms[i]]*1e-3 for i in range(len(ms) - 1)))
fine = [(100, 800), (120, 960), (140, 840)]
e100 = abs(P_from_u(sols[(100, 800)]) - ref)
e120 = abs(P_from_u(sols[(120, 960)]) - ref)
check('the 256-bit solutions converge: |P(m) - P(140)| = %s (m = 100), %s (m = 120), a factor %s per 20 Chebyshev terms' %
      (mp.nstr(e100, 3), mp.nstr(e120, 3), mp.nstr(e100/e120, 2)), e100/e120 > 1e4 and e120 < 2*mp.mpf(10)**-40)
check('so the m = 140 value, expected to be within about %s of the limit if that rate persists, gives P_inf = %s to 40 digits' %
      (mp.nstr(e120**2/e100, 1), P40), mp.nstr(ref, 40) == P40 and e120**2/e100 < mp.mpf(10)**-42)
say('\n2. The two finest recomputed in Arb at 256 bits  (%.0f s)' % (time.time() - T0))
for key in ((140, 840), (120, 960)):
    d = sols[key]
    H = HP(d['m'], d['n'], prec=256)
    u = [arb(x) for x in d['u']]
    F = H.F(u)
    r = max(abs(float(x.mid())) for x in F)
    gs = float((H.last['Gsheet'] - 2 - arb(2).sqrt()).mid())
    I = float(H.last['I'].mid())
    Pa = P_of(u, d['m'])
    check('m = %d, n = %d: largest collocation residual %.1e; sheet circulation - (2 + sqrt2) = %.1e; angular impulse %.1e; P = %s' %
          (key[0], key[1], r, gs, I, Pa.mid().str(42, radius=False)), r < (1e-38 if key[0] == 140 else 1e-32) and abs(gs) < 1e-35 and abs(I) < 1e-35)

say('\n3. The sheet circulation from sum_{i<j} Gamma_i Gamma_j = 0')
Ss = sp.Symbol('S', positive=True)
cond = 1 + 2*(-1)*Ss + Ss**2/2
roots = sp.solve(sp.Eq(cond, 0), Ss)
check('two vortices -1 and a sheet of circulation S (self-pairs S^2/2 in the continuum): 1 - 2S + S^2/2 = 0 has the roots %s; the solution '
      'has the larger, 2 + sqrt2, and each arm carries 1 + 1/sqrt2' % roots, set(roots) == {2 - sp.sqrt(2), 2 + sp.sqrt(2)})

say('\n4. Independent residual of the continuous equations between collocation points  (%.0f s)' % (time.time() - T0))
d = sols[(140, 840)]
m = d['m']
u = np.array([float(x) for x in d['u']])
a, b, L, zn, lam = u[:m], u[m:2*m], u[2*m], u[2*m + 1] + 1j*u[2*m + 2], u[2*m + 3] + 1j*u[2*m + 4]
ca = np.zeros(2*m - 1)
ca[::2] = a
cb = np.zeros(2*m - 1)
cb[::2] = b
K = 400
ce = C.chebinterpolate(lambda x: np.exp(1j*C.chebval(x, ca)).real, K) + 1j*C.chebinterpolate(lambda x: np.exp(1j*C.chebval(x, ca)).imag, K)
cz = C.chebint(ce, lbnd=0)*L
czp = C.chebder(cz)
mp.mp.dps = 40
czm = [mp.mpc(c.real, c.imag) for c in cz]
czpm = [mp.mpc(c.real, c.imag) for c in czp]
cbm = [mp.mpf(c) for c in cb]
znm, lamm = mp.mpc(zn.real, zn.imag), mp.mpc(lam.real, lam.imag)


def clen(c, x):
    b1 = b2 = 0
    for ck in reversed(c[1:]):
        b1, b2 = 2*x*b1 - b2 + ck, b1
    return x*b1 - b2 + c[0]


def zf(x):
    return clen(czm, x)


def gf(x):
    return mp.sqrt(1 - x*x)*clen(cbm, x)


def Fsheet(x0):
    x0 = mp.mpf(x0)
    z0 = zf(x0)
    A = gf(x0)/clen(czpm, x0)

    def f(t):
        return gf(t)/(z0 - zf(t)) + A/(t - x0)
    p0 = mp.acos(x0)
    Iv = mp.quad(lambda p: f(mp.cos(p))*mp.sin(p) if mp.cos(p) != x0 else 0, [0, p0, mp.pi], method='gauss-legendre') - A*mp.log((1 - x0)/(1 + x0))
    return Iv - 1/(z0 - znm) - 1/(z0 + znm) - lamm*mp.conj(z0)


worstF = 0
for x0 in ('0.0137', '0.3', '0.77', '0.99', '0.9995'):
    r = abs(Fsheet(x0))
    worstF = max(worstF, r)
    say('   sheet at x = %s: |residual| = %s' % (x0, mp.nstr(r, 3)))
wn = mp.quad(lambda p: gf(mp.cos(p))*mp.sin(p)/(znm - zf(mp.cos(p))), [0, mp.pi/2, mp.pi], method='gauss-legendre') - 1/(2*znm) - lamm*mp.conj(znm)
say('   point vortex: |residual| = %s' % mp.nstr(abs(wn), 3))
check('the continuous equations hold between the collocation points to %s (sheet) and %s (vortex), at the binary64 floor of this check' %
      (mp.nstr(worstF, 2), mp.nstr(abs(wn), 2)), worstF < 1e-12 and abs(wn) < 1e-12)

say('\n5. Local minimality along the family (binary64, m = 40, n = 240)  (%.0f s)' % (time.time() - T0))
M140 = Model(140, 840)
M = Model(40, 240)
u40 = M140.resize(u, 40)
u40, r40 = gn(M, u40)
n0, s = nullvec(M, u40)
c0 = n0 @ u40
hstep = 1e-3
Ps = []
for dc in (-hstep, 0.0, hstep):
    w, rr = solve_plane(M, u40 + dc*n0, n0, c0 + dc)
    Ps.append(M.P(w))
d1 = (Ps[2] - Ps[0])/(2*hstep)
d2P = (Ps[2] - 2*Ps[1] + Ps[0])/hstep**2
check('the discrete solution set is one-dimensional (singular values / largest: %.1e, then %.1e), and along it dP/dc = %.1e, d^2P/dc^2 = %.4f > 0: '
      'a local minimum; P = %.14f at m = 40' % (s[-1]/s[0], s[-2]/s[0], d1, d2P, Ps[1]),
      s[-1]/s[0] < 1e-6 and s[-2]/s[0] > 1e-5 and abs(d1) < 1e-6 and d2P > 0.05 and abs(Ps[1] - float(P40)) < 1e-11)

say('\n6. The finite two-arm family of the companion paper  (%.0f s)' % (time.time() - T0))
tw = json.load(open(os.path.join(DATA, 'twoarm-family.json')))
Nn = np.array(tw['N'], float)
Pn = np.array([float(x) for x in tw['P']])
sel = Nn >= 301
A = np.stack([1/Nn[sel]**i for i in range(5)], 1)
coef, *_ = np.linalg.lstsq(A, Pn[sel], rcond=None)
rms = np.sqrt(np.mean((A @ coef - Pn[sel])**2))
A2 = A[:, 1:]
coef2, *_ = np.linalg.lstsq(A2, Pn[sel] - float(P40), rcond=None)
rms2 = np.sqrt(np.mean((A2 @ coef2 - (Pn[sel] - float(P40)))**2))
pts = [101, 203, 403, 603]
mp.mp.dps = 30
xs = [mp.mpf(1)/N for N in pts]
ys = [mp.mpf(tw['P'][tw['N'].index(N)]) for N in pts]
rich = mp.fsum(ys[i]*mp.fprod((0 - xs[j])/(xs[i] - xs[j]) for j in range(4) if j != i) for i in range(4))
check('fit P = a + b/N + ... + e/N^4 over N >= 301 (%d members, P stored to 13 digits): a = %.13f (rms %.1e), within %.1e of P_inf; '
      'with a fixed at P_inf the rms stays %.1e' % (sel.sum(), coef[0], rms, abs(coef[0] - float(P40)), rms2),
      abs(coef[0] - float(P40)) < 1e-11 and rms2 < 2*rms)
check('cubic Richardson extrapolation in 1/N through N = 101, 203, 403, 603 gives %s, within %s of P_inf' % (mp.nstr(rich, 9), mp.nstr(abs(rich - mp.mpf(P40)), 2)),
      abs(rich - mp.mpf(P40)) < 1e-6)

say('\n7. Shape of the solution (m = 140)')
xs_ = np.linspace(0, 1, 2001)
hmin = min(C.chebval(x, cb) for x in xs_)
gam = np.array([np.sqrt(max(0.0, 1 - x*x))*C.chebval(x, cb) for x in xs_])/L
arm = np.trapezoid(np.sqrt(np.clip(1 - xs_**2, 0, None))*C.chebval(xs_, cb), xs_) if hasattr(np, 'trapezoid') else None
from scipy.integrate import quad  # noqa: E402
armq = quad(lambda x: np.sqrt(max(0.0, 1 - x*x))*C.chebval(x, cb), 0, 1, epsabs=1e-14, limit=200)[0]
th = C.chebval(xs_, ca)
say('   tip z(1) = %s; half arc length L = %.10f; vortices at +-(%.10f %+.10fi), |z_n| = %.8f; Lambda = %.10f %+.10fi; P = -Re Lambda/(2 Im Lambda) = %.15f' %
    (np.round(C.chebval(1.0, cz), 12), L, zn.real, zn.imag, abs(zn), lam.real, lam.imag, -lam.real/(2*lam.imag)))
say('   density per unit length: %.6f at the centre, largest %.6f at s/L = %.3f; tangent angle %.2f deg at the centre, %.2f deg at the tip' %
    (gam[0], gam.max(), xs_[gam.argmax()], np.degrees(th[0]), np.degrees(th[-1])))
check('the sheet density is positive (least h = %.6f > 0) and each arm carries %.12f = 1 + 1/sqrt2 (difference %.1e)' %
      (hmin, armq, armq - 1 - 1/np.sqrt(2)), hmin > 0 and abs(armq - 1 - 1/np.sqrt(2)) < 1e-10)

say('\n8. No closed form found  (%.0f s)' % (time.time() - T0))
mp.mp.dps = 45
P = mp.mpf(P40)
tol = mp.mpf(10)**-37
found = []
for name, x in [('P', P), ('P^2', P**2), ('1/P', 1/P), ('arctan(2P)/pi', mp.atan(2*P)/mp.pi), ('4P^2 + 1', 4*P**2 + 1)]:
    for deg in range(1, 6):
        p = mp.findpoly(x, deg, maxcoeff=10**6, tol=tol)
        if p:
            found.append((name, deg, p))
            break
check('no algebraic relation of degree <= 5 with integer coefficients <= 1e6 for P, P^2, 1/P, arctan(2P)/pi or 4P^2 + 1 '
      '(with 40 digits a chance relation of degree d needs coefficients near 10^(40/(d+1)), above 1e6 for d <= 5)', not found, str(found) if found else '')
Cst = {'pi': mp.pi, 'sqrt2': mp.sqrt(2), 'sqrt3': mp.sqrt(3), 'log2': mp.log(2), 'log3': mp.log(3), 'e': mp.e,
       'log(1+sqrt2)': mp.log(1 + mp.sqrt(2))}
bases = [['sqrt2', 'sqrt3', 'pi'], ['sqrt2', 'pi', 'log2', 'log(1+sqrt2)'], ['sqrt3', 'pi', 'log2', 'log3'], ['e', 'pi', 'sqrt2', 'log2']]
rels = []
for bs in bases:
    for name, x in [('P', P), ('P^2', P**2), ('1/P', 1/P)]:
        rel = mp.pslq([x, mp.mpf(1)] + [Cst[c] for c in bs], tol=tol, maxcoeff=10**6, maxsteps=10**6)
        if rel:
            rels.append((name, bs, rel))
check('PSLQ finds no integer relation with coefficients <= 1e6 between P (or P^2, 1/P), 1 and the constants of %d bases of at most four '
      'constants (%s); a chance relation among six numbers needs coefficients near 10^(40/6), above 1e6' % (len(bases), '; '.join(','.join(b) for b in bases)), not rels, str(rels) if rels else '')

say('\n%d checks, %d failed  (%.0f s)' % (NCHK[0], len(FAILED), time.time() - T0))
with open(os.path.join(DATA, 'verify-continuum-limit.txt'), 'w') as fh:
    fh.write('\n'.join(OUT) + '\n')
sys.exit(1 if FAILED else 0)
