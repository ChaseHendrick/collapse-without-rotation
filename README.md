# Point-Vortex Collapse Without Rotation: A Cluster Mechanism, a Phase Diagram and a Continuum Limit

**Chase Hendrick**, Independent Researcher · [ORCID 0009-0002-9754-6087](https://orcid.org/0009-0002-9754-6087)

**Preprint**, not peer reviewed. It is the sequel to *Minimal Winding in the Self-Similar Collapse of Point Vortices*
([ChaseHendrick/minimal-winding](https://github.com/ChaseHendrick/minimal-winding)), which it cites as its companion.

**[Read the preprint (PDF, 24 pages)](paper/collapse-without-rotation.pdf)**

## Abstract

In a self-similar collapse of point vortices the configuration turns through the angle $P$ while the square of its size
decreases by the factor $e$. A companion paper proved that $P > \sqrt{3}/2$ for three Euler vortices, that the same
constant is the limit for a strong vortex carrying weak tight pairs of opposite sign, and, with computer-assisted
proofs, that eleven vortices in the $\alpha = 2$ model and sixty in the surface quasi-geostrophic (SQG) model can
collapse without rotating at all. We first show that the constant $\sqrt{3}/2$ belongs to a larger class: a strong Euler
vortex carrying any number of weak tight clusters, of any sizes and signs, has $P \ge \sqrt{3}/2 - o(1)$ as the clusters
weaken. If $P$ stays bounded, every cluster is close to a translating relative equilibrium whose net circulation is half
the sum of the squares of its circulations, measured in units of the strong one; a single isolated weak vortex forces
$P$ to grow like the inverse of the weak circulations, with an explicit constant. In particular no such configuration
collapses without rotation, whatever the number of vortices. We prove that a weak triple of signs $(+,+,-)$ beside a
positive strong vortex gives collapses with $P$ below $\sqrt{3}/2$ by an amount proportional to the weak circulations,
so that $\sqrt{3}/2$ is a limit and not a bound at a fixed strength; a formal expansion, confirmed by high-precision
numerical collapses computed at fifty digits, extends the first-order term to clusters of any size and explains why four
vortices are the first to go below $\sqrt{3}/2$. We then map, numerically, the least $\alpha$ at which $N$ vortices can
collapse without rotation along two families: the least $N$ is $5$, $6$, $8$, $11$, $17$, $29$ and $60$ for $\alpha =
6$, $4$, $3$, $2$, $1.5$, $1.2$ and $1$, and fits of the thresholds give a positive limit between $0.66$ and $0.80$,
while fits forced to the Euler value $0$ are much worse. No self-similar collapse with nonzero total circulation is
mirror symmetric; numerically, the two certified collapses without rotation are linearly unstable, with $8$ and $57$
unstable modes, and perturbed motions leave them without turning. Finally, numerically, the Euler minimizers of the
companion paper approach a continuum of two point vortices and a positive vortex sheet, a kind of collapse that O'Neil
found with point vortices approximating the sheet; with a density that vanishes like a square root at the tips of the
sheet, the winding has a local minimum along a one-parameter family of such continua, $P_\infty =
0.47736353369161202484\ldots$, stable to forty digits if the observed exponential convergence persists, which agrees
with the extrapolation of the finite family.

## Status of the results

- **Proved:** Theorem 1 (a strong vortex with weak tight clusters; its part (d) with explicit constants),
  Corollary 1 (no collapse without rotation in that regime, for any number of vortices), Theorem 2 and Corollary 2
  (one weak triple: a family of collapses with $P = \sqrt3/2 + (2\sqrt3\,g_1g_2g_3/S)\gamma + O(\gamma^2)$, so for
  signs $(+,+,-)$ the constant is undershot at every small strength; the first-order coefficient is computed exactly
  by computer algebra), Theorem 3 (the same family for any single weak cluster that translates nondegenerately),
  Lemma 1 (such clusters exist for every number of vortices, built from a pair or the triangle (1, 1, -2) by adding
  weak vortices +-eps at stagnation points), Corollary 3 (the constant in Theorem 1(a) is sharp for every number of
  weak vortices, for small enough c), Proposition 1 (no self-similar collapse with nonzero total circulation is mirror
  symmetric).
- **Formal, confirmed numerically:** the next-order expansion of Section 3.3 for several clusters and for clusters of
  more than three vortices.
- **Numerical:** the phase diagram (Section 4), the family and its stability (Sections 5.2 and 5.3), the continuum
  limit (Section 6).
- **Pending before this draft can advance:** nothing in the text. O'Neil, *Collapse and concentration of vortex sheets
  in two-dimensional flow*, Theor. Comput. Fluid Dyn. 24 (2010) 39-44, has been read (2026-09-25; the owner supplied
  the PDF, which is not committed): he found collapsing sheets with point vortices first, including the S-shaped
  sheet between two negative point vortices at P = 1/2, and Section 6 credits him; the minimum P_inf along the family
  and the direct continuum solution are not in his paper. A second reader has checked the draft (2026-09-25) and
  its corrections are in. Aref, Newton, Stremler, Tokieda and Vainchtein, *Vortex crystals* (2003), has been read for the construction of
  Lemma 1: it gives the translating equilateral triangle and, for circulations of equal magnitude, the restriction to
  triangular numbers, both now cited, and not the construction. O'Neil, Nonlinearity 26 (2013), is read in summary
  only (paywalled), as the paper says.

## Contents

| Folder | What is in it |
|---|---|
| [`paper/`](paper/) | The manuscript: [`collapse-without-rotation.tex`](paper/collapse-without-rotation.tex) (LaTeX, the only source), its build [`collapse-without-rotation.pdf`](paper/collapse-without-rotation.pdf), and [`figures/`](paper/figures/) |
| [`code/`](code/) | The programs below, their shared modules and [`requirements.txt`](code/requirements.txt) |
| [`data/`](data/) | The output of the programs, the stored configurations of the phase diagram (`phase-diagram/`), the searches (`searches/`) and the continuum solutions (`continuum/`), and the inputs copied from the companion paper's data: `collapse-alpha2-n11-no-rotation.json`, `collapse-sqg-n60-no-rotation.json`, `collapse-euler-n4-minimum.json`, `collapse-euler-n33.json`, `collapse-euler-n61.json`, `collapse-euler-n603.json` and `twoarm-family.json` |

| Program | What it checks | Checks | Time |
|---|---|---:|---|
| [`verify_cluster_identities.py`](code/verify_cluster_identities.py) | Every identity in the proof of Theorem 1 and in the formal expansion, exactly (SymPy), with negative controls | 62 | 10 s |
| [`verify_cluster_remainders.py`](code/verify_cluster_remainders.py) | The remainders of the proof on random configurations, $\gamma$ from $10^{-2}$ to $10^{-7}$, a control that must grow, and the explicit constants of Step 1 and part (d) | 8 | 3 s |
| [`verify_cluster_collapses.py`](code/verify_cluster_collapses.py) | Collapses computed at 50 digits of a strong vortex with clusters (Table 1), with convergence tests; singletons; controls. `--quick`: 72 checks, one minute | 90 | 2 min |
| [`verify_triple_branch.py`](code/verify_triple_branch.py) | Theorem 2 and Corollary 2: the point at $\gamma = 0$, the closed form of $d(w_1 - w_2)$ and the slope $2\sqrt3\,g_1g_2g_3/S$, exactly over $\mathbb{Q}(t)(\omega)$; the family at 40 digits against the Biot-Savart equations; the class $K(3, 1/2)$; a triple with $g_1g_2g_3 > 0$; controls | 62 | 10 s |
| [`verify_sharpness_all_n.py`](code/verify_sharpness_all_n.py) | Lemma 1 and Corollary 3: the pair and the triangle exactly (translation, nondegeneracy, the zeros of R, discriminant -216); clusters of 4 to 15 vortices grown by the induction from both starting clusters (binary64) and their collapse families at forty digits, checked against the Biot-Savart equations of all vortices; controls; with `--large`, a cluster of 101 vortices | 37 (40 with `--large`) | 20 s (`--large`: 1 min) |
| [`verify_cluster_stored.py`](code/verify_cluster_stored.py) | The 21 stored collapses rechecked with separate code at 60 digits | 26 | 1 s |
| [`verify_phase_diagram.py`](code/verify_phase_diagram.py) | All 120 thresholds from their stored configurations, Tables 2 and 3, the searches | 46 | 15 s |
| [`plot_phase_diagram.py`](code/plot_phase_diagram.py) | Figure 1 | | 3 s |
| [`verify_family_structure.py`](code/verify_family_structure.py) | The nine-parameter family at $\alpha = 2$, mirror images, the ring reduction | 15 | 40 s |
| [`verify_stability.py`](code/verify_stability.py) | Exponents and direct integrations (Table 4). `--quick` (SQG integrations to $0.08\,t_c$): 13 checks, 20 s | 14 | 2.5 min |
| [`verify_continuum_limit.py`](code/verify_continuum_limit.py) | The continuum limit (Table 5): convergence, Arb recheck, residual between collocation points, local minimality, the finite family, closed-form searches | 13 | 75 s |
| [`continuum_arb.py`](code/continuum_arb.py) | Also a program: refines a continuum solution at a chosen resolution (default $m = 64$, $n = 384$, about a minute) | | 1 min |

The modules [`collapse_core.py`](code/collapse_core.py) (the $\alpha$-model law, its Jacobian, Newton's method and
the first-order system of the minimization of $\alpha$), [`continuum_model.py`](code/continuum_model.py) and
[`continuum_arb.py`](code/continuum_arb.py) are shared by the programs.

## Reproduce

From this folder:

```
python3 -m pip install -r code/requirements.txt
python3 code/verify_cluster_identities.py
python3 code/verify_cluster_remainders.py
python3 code/verify_cluster_collapses.py
python3 code/verify_triple_branch.py
python3 code/verify_sharpness_all_n.py
python3 code/verify_cluster_stored.py
python3 code/verify_phase_diagram.py
python3 code/plot_phase_diagram.py
python3 code/verify_family_structure.py
python3 code/verify_stability.py
python3 code/verify_continuum_limit.py
cd paper && pdflatex collapse-without-rotation.tex && pdflatex collapse-without-rotation.tex && pdflatex collapse-without-rotation.tex
```

Each verification program exits with an error if any check fails and writes its report to `data/`.

## Cite

Until the paper has a DOI of its own:

```bibtex
@misc{hendrick2026collapse,
  author = {Hendrick, Chase},
  title  = {Point-Vortex Collapse Without Rotation: A Cluster Mechanism, a Phase Diagram and a Continuum Limit},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/ChaseHendrick/collapse-without-rotation}
}
```

## License

The manuscript in `paper/`, its figures included, is Copyright (c) 2026 Chase Hendrick, all rights reserved. The
programs in `code/` and the data in `data/` are under the Apache License 2.0. The `LICENSE` file has both.
