# Releases

Each release of this repository is archived on Zenodo with its own DOI. The manuscript is a preprint and has not been
peer reviewed.

## 1.0.0 (2026-09-25)

**DOI:** [10.5281/zenodo.22969841](https://doi.org/10.5281/zenodo.22969841)

The first public release of the preprint *Point-Vortex Collapse Without Rotation: A Cluster Mechanism, a Phase
Diagram and a Continuum Limit*, with the programs that check its results and their output. It is the sequel to
*Minimal Winding in the Self-Similar Collapse of Point Vortices* (programs and data: doi:10.5281/zenodo.22963796).

### What the paper shows

In a self-similar collapse of point vortices the configuration turns through the angle P while the square of its size
decreases by the factor e. For three Euler vortices P > √3/2 (the companion paper). This paper asks when many vortices
can collapse turning less, or not at all.

- **Proved: weak clusters cannot stop the rotation.** A strong vortex carrying weak, tight clusters of any sizes and
  signs has P ≥ √3/2 − o(1) as the clusters weaken (Theorem 1), so no such configuration collapses without rotation,
  whatever the number of vortices (Corollary 1). A single isolated weak vortex forces P to grow like the inverse of the
  weak circulations, with an explicit constant.
- **Proved: √3/2 is a limit, not a bound.** One weak triple of signs (+, +, −) gives a family of collapses with
  P = √3/2 + (2√3 g₁g₂g₃/S)γ + O(γ²), below √3/2 at every small strength (Theorem 2 and Corollary 2; the first-order
  coefficient is computed exactly by computer algebra).
- **Proved: the constant is sharp for every number of weak vortices.** Nondegenerate translating clusters exist for
  every number of vortices, built from a pair or the triangle (1, 1, −2) by adding weak vortices ±ε at simple
  stagnation points (Lemma 1); each gives a family of collapses with P → √3/2 (Theorem 3 and Corollary 3).
- **Proved:** no self-similar collapse with nonzero total circulation is mirror symmetric (Proposition 1).
- **Numerical, labelled as such:** the least α at which N vortices collapse without rotation in the α-models (least N
  5, 6, 8, 11, 17, 29 and 60 for α = 6, 4, 3, 2, 1.5, 1.2 and 1, with fits pointing to a positive limit between 0.66
  and 0.80 rather than the Euler value 0); the nine-parameter family of collapses without rotation of eleven vortices
  at α = 2 and the instability of both certified collapses without rotation (8 and 57 unstable modes); and a continuum
  limit of two point vortices and a vortex sheet, along whose family the winding has a local minimum
  P∞ = 0.47736353369161202484…. O'Neil (Theor. Comput. Fluid Dyn. 24, 2010) found collapsing sheets with point vortices
  first, and the paper credits him.

### Checked by computer

Ten verification programs, 373 checks in all (376 with `--large`). Each stops with an error if any check fails and
writes its report to `data/`.

- `code/verify_cluster_identities.py`: every identity in the proof of Theorem 1 and in the formal expansion, exactly
  (SymPy), with negative controls; 62 checks.
- `code/verify_cluster_remainders.py`: the remainders of the proof and the explicit constants; 8 checks.
- `code/verify_cluster_collapses.py`: collapses at 50 digits of a strong vortex with clusters (Table 1); 90 checks.
- `code/verify_triple_branch.py`: Theorem 2 and Corollary 2, exactly and at 40 digits; 62 checks.
- `code/verify_sharpness_all_n.py`: Lemma 1 and Corollary 3, the starting clusters exactly, clusters of 4 to 15
  vortices and their collapse families at 40 digits; 37 checks (40 with `--large`, a cluster of 101 vortices).
- `code/verify_cluster_stored.py`: the stored collapses rechecked with separate code at 60 digits; 26 checks.
- `code/verify_phase_diagram.py`: the 120 thresholds of the phase diagram and Tables 2 and 3; 46 checks.
- `code/verify_family_structure.py`: the family of collapses without rotation at α = 2; 15 checks.
- `code/verify_stability.py`: stability exponents and direct integrations (Table 4); 14 checks.
- `code/verify_continuum_limit.py`: the continuum limit (Table 5), with a recheck in ball arithmetic; 13 checks.

### Files

- `paper/collapse-without-rotation.pdf`: the paper. `paper/collapse-without-rotation.tex` is its LaTeX source and
  `paper/figures/` holds the figure.
- `code/`: the verification programs, their shared modules (`collapse_core.py`, `continuum_model.py`,
  `continuum_arb.py`), the plotting script and `requirements.txt`.
- `data/`: the output of the programs, the stored configurations of the phase diagram, the searches, the continuum
  solutions, and the inputs copied from the companion paper's data.

### Reproduce

```
python3 -m pip install -r code/requirements.txt
python3 code/verify_cluster_identities.py
python3 code/verify_cluster_remainders.py
python3 code/verify_cluster_collapses.py
python3 code/verify_triple_branch.py
python3 code/verify_sharpness_all_n.py
python3 code/verify_cluster_stored.py
python3 code/verify_phase_diagram.py
python3 code/verify_family_structure.py
python3 code/verify_stability.py
python3 code/verify_continuum_limit.py
```

### License

The manuscript in `paper/`, its figures included, is Copyright (c) 2026 Chase Hendrick, all rights reserved. The
programs in `code/` and the data in `data/` are licensed under the Apache License 2.0.
