# surfacedispersion-python

A Python port of the Mathematica notebook `surfacedispersion.nb`.

It demonstrates:
- Numerical eigendecomposition of a concrete 4x4 real symmetric matrix `M`
- Symmetry/commutation checks with a signature (indefinite metric) matrix `U = diag(1, -1, 1, -1)`
- Symbolic computation of the **centralizer** of `M` (all matrices `P` that commute with `M`, i.e., `P M = M P`)
- Construction of a symbolic complex 4x4 matrix `H` that is **pseudo‑Hermitian** with respect to `U` (`H† U = U H` ⇔ `H† = U H U`)
- A minimal 2x2 toy example showing non‑commutation with a sign‑flip

The repo contains a small library (`surfacedispersion`) and a CLI demo (`run_demo.py`) that reproduces the notebook computations end‑to‑end.

## Quickstart

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install
**Option A (requirements):**
```bash
pip install -r requirements.txt
```

**Option B (pyproject):**
```bash
pip install .
```

### 3) Run the demo script
```bash
python run_demo.py
```

This prints:
- Eigenvalues/eigenvectors of `M`
- Commutation tests with `U`
- Centralizer basis dimension and representative basis elements
- Pseudo‑Hermiticity checks for `H`
- The 2x2 toy example

## Library usage

```python
import numpy as np
from surfacedispersion import (
    default_M, signature_U, eigensystem_numeric,
    centralizer_symbolic, build_symbolic_H, check_pseudo_hermiticity
)

M = default_M()
U = signature_U()

# 1) Numeric eigensystem
evals, evecs, diag_via_Q = eigensystem_numeric(M)

# 2) Centralizer (symbolic)
sol, free_params, centralizer_basis, poly_basis_rank = centralizer_symbolic(M)

# 3) Pseudo‑Hermitian H and checks
r,w,z,v,x,y = 1,2,3,4,5,6  # sample real parameters
H = build_symbolic_H(r,w,z,v,x,y)
ok1, ok2 = check_pseudo_hermiticity(H, U)
```

## Files

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── run_demo.py
├── src/
│   └── surfacedispersion/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_core.py
```


---

## `001surface.nb` (surface model) — Python port

This repo also includes a port of the structured, symbolic **surface model** from `001surface.nb`:

- Builds a parametric complex 4×4 matrix `H(r,w,z,v,x,y)` with a block/structured pattern
- Verifies **pseudo-Hermiticity** of `H` with respect to `U = diag(1,-1,1,-1)`

### Run the surface-model demo
```bash
python run_surface.py
```

### Programmatic use
```python
import sympy as sp
from surfacedispersion import surface_signature_U, build_block_surface_H, is_pseudo_hermitian

U = surface_signature_U()
H = build_block_surface_H(1,2,3,4,5,6)
ok1, ok2 = is_pseudo_hermitian(H, U)
```
