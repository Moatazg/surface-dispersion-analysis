import sympy as sp

def signature_U_4():
    """Return the 4x4 signature (indefinite metric) matrix U = diag(1,-1,1,-1) as a SymPy Matrix."""
    return sp.diag(1, -1, 1, -1)

def build_block_surface_H(r, w, z, v, x, y):
    r"""
    Build a 4x4 complex, structured (block-like) matrix H parameterized by real scalars r,w,z,v,x,y.

    This mirrors the structure commonly used in pseudo-Hermitian / PT-symmetric toy models,
    and matches the pattern in `001surface.nb`:
        - purely imaginary (i * w * z) entries coupling (1,2) and (3,4)
        - real/imaginary mixture -v*(i*x - y) and -v*(-i*x - y) coupling (1,4) and (2,3)
        - diagonal [ r, -r, r, -r ] pattern

    Returns
    -------
    H : sympy.Matrix (4x4, complex)
    """
    I = sp.I
    return sp.Matrix([
        [ r,             I*w*z,                 0,             -v*(I*x - y) ],
        [-I*w*z,         -r,           -v*(-I*x - y),                   0   ],
        [ 0,             -v*(I*x - y),         r,               I*w*z       ],
        [-v*(-I*x - y),   0,                 -I*w*z,                  -r     ]
    ])

def is_pseudo_hermitian(H: sp.Matrix, U: sp.Matrix | None = None):
    """
    Check pseudo-Hermiticity with respect to U:
        H† U = U H    and    H† = U H U
    Returns a tuple of booleans (ok1, ok2).
    """
    if U is None:
        U = signature_U_4()
    Hdag = H.conjugate().T
    left  = sp.simplify(Hdag*U - U*H)
    right = sp.simplify(Hdag - U*H*U)
    return (left == sp.zeros(4), right == sp.zeros(4))

def surface_model_summary(r=1, w=2, z=3, v=4, x=5, y=6):
    """
    Convenience: build H with sample parameters, test pseudo-Hermiticity, and return a dict summary.
    """
    U = signature_U_4()
    H = build_block_surface_H(r,w,z,v,x,y)
    ok1, ok2 = is_pseudo_hermitian(H, U)
    return {
        "H": H,
        "U": U,
        "pseudo_hermitian_HdagU_eq_UH": ok1,
        "pseudo_hermitian_Hdag_eq_UHU": ok2,
    }
