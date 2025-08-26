import numpy as np
import sympy as sp

def default_M() -> np.ndarray:
    """Return the concrete 4x4 real symmetric matrix M from the notebook."""
    return np.array([
        [-1.26,  0.71, -0.42, -0.15],
        [ 0.71,  0.30, -1.42,  0.40],
        [-0.42, -1.42,  1.58,  0.75],
        [-0.15,  0.40,  0.75,  0.76]
    ], dtype=float)

def signature_U() -> np.ndarray:
    """Return U = diag(1, -1, 1, -1)."""
    return np.diag([1, -1, 1, -1]).astype(float)

def eigensystem_numeric(M: np.ndarray):
    """
    Compute eigenvalues/eigenvectors for symmetric M with numpy, and Q^T M Q.
    Returns (evals, evecs, diag_via_Q).
    """
    evals, evecs = np.linalg.eigh(M)
    diag_via_Q = evecs.T @ M @ evecs
    return evals, evecs, diag_via_Q

def centralizer_symbolic(M: np.ndarray):
    """
    Solve P M = M P symbolically with SymPy.

    Returns:
      sol: SymPy Matrix (4x4) whose entries are linear expressions in free parameters
      free_params: list of SymPy symbols parameterizing the general solution
      centralizer_basis: list of basis matrices obtained by setting one free param to 1 at a time
      poly_basis_rank: rank of {I, M, M^2, M^3} (should be 4 for simple spectrum)
    """
    Ms = sp.nsimplify(sp.Matrix(M))

    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p = sp.symbols('a b c d e f g h i j k l m n o p', real=True)
    P = sp.Matrix(4, 4, [a,b,c,d, e,f,g,h, i,j,k,l, m,n,o,p])

    comm = P*Ms - Ms*P
    eqs = list(comm)
    vars_ = (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p)

    sol_tuple = list(sp.linsolve(eqs, vars_))[0]
    free_params = sorted(list({s for s in sol_tuple.free_symbols}), key=lambda s: s.name)
    P_sol = sp.Matrix(4, 4, list(sol_tuple))

    # Give a basis of solutions by toggling free parameters
    centralizer_basis = []
    for fp in free_params:
        subs_map = {t: (1 if t == fp else 0) for t in free_params}
        centralizer_basis.append(sp.simplify(P_sol.subs(subs_map)))

    # Polynomial basis {I, M, M^2, M^3}
    I4 = sp.eye(4)
    B = [I4, Ms, Ms**2, Ms**3]
    Bcols = [sp.Matrix(b).reshape(16,1) for b in B]
    rank_B = sp.Matrix.hstack(*Bcols).rank()

    return P_sol, free_params, centralizer_basis, rank_B

def build_symbolic_H(r, w, z, v, x, y):
    """
    Build the symbolic complex 4x4 matrix H whose structure is pseudo-Hermitian w.r.t U.
    All parameters are assumed real.
    """
    I = sp.I
    H = sp.Matrix([
        [ r,             I*w*z,                 0,             -v*(I*x - y) ],
        [-I*w*z,         -r,           -v*(-I*x - y),                   0   ],
        [ 0,             -v*(I*x - y),         r,               I*w*z       ],
        [-v*(-I*x - y),   0,                 -I*w*z,                  -r     ]
    ])
    return H

def check_pseudo_hermiticity(H: sp.Matrix, U=None):
    """
    Check H† U = U H and H† = U H U (both should be True).
    If U is None, use diag(1,-1,1,-1).
    Returns (ok1, ok2).
    """
    if U is None:
        U = sp.diag(1, -1, 1, -1)
    else:
        U = sp.Matrix(U) if not isinstance(U, sp.Matrix) else U

    H_dag = H.conjugate().T
    left  = sp.simplify(H_dag*U - U*H)
    right = sp.simplify(H_dag - U*H*U)
    return (left == sp.zeros(4), right == sp.zeros(4))

def toy_2x2_noncommutation():
    """
    Return the 2x2 toy example matrices (A2, U2) and their commutator.
    """
    A2 = sp.Matrix([[1.95, -0.64],
                    [-0.64, 0.10]])
    U2 = sp.diag(1, -1)
    comm = A2*U2 - U2*A2
    return A2, U2, comm
