import sympy as sp
from surfacedispersion import surface_signature_U, build_block_surface_H, is_pseudo_hermitian

def test_surface_pseudo_hermiticity():
    U = surface_signature_U()
    H = build_block_surface_H(1,2,3,4,5,6)
    ok1, ok2 = is_pseudo_hermitian(H, U)
    assert ok1 and ok2
