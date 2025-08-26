import numpy as np
import sympy as sp
from surfacedispersion import (
    default_M, signature_U, eigensystem_numeric,
    centralizer_symbolic, build_symbolic_H, check_pseudo_hermiticity
)

def test_eigensystem_shapes():
    M = default_M()
    evals, evecs, diag_via_Q = eigensystem_numeric(M)
    assert evals.shape == (4,)
    assert evecs.shape == (4,4)
    assert diag_via_Q.shape == (4,4)

def test_trace_matches_sum_eigs():
    M = default_M()
    evals, _, _ = eigensystem_numeric(M)
    assert np.isclose(np.trace(M), evals.sum())

def test_centralizer_rank():
    M = default_M()
    _, free_params, _, rank_poly = centralizer_symbolic(M)
    assert len(free_params) == 4
    assert rank_poly == 4

def test_pseudo_hermiticity_identity():
    U = sp.diag(1, -1, 1, -1)
    H = build_symbolic_H(1,2,3,4,5,6)
    ok1, ok2 = check_pseudo_hermiticity(H, U)
    assert ok1 and ok2
