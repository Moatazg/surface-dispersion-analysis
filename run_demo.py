#!/usr/bin/env python3
import numpy as np
import sympy as sp
from surfacedispersion import (
    default_M, signature_U, eigensystem_numeric,
    centralizer_symbolic, build_symbolic_H, check_pseudo_hermiticity,
    toy_2x2_noncommutation
)

np.set_printoptions(precision=6, suppress=True, linewidth=120)

def main():
    print("=== surfacedispersion demo ===")

    # 1) Numeric eigensystem
    M = default_M()
    print("\n[1] Matrix M:\n", M)
    evals, evecs, diag_via_Q = eigensystem_numeric(M)
    print("\nEigenvalues =", evals)
    print("\nEigenvectors (columns):\n", evecs)
    print("\nQ^T M Q:\n", diag_via_Q)
    print("\nTrace(M) vs sum(eigs):", np.trace(M), evals.sum())

    # 2) Signature matrix U and commutation
    U = signature_U()
    MU = M @ U
    UM = U @ M
    frob = np.linalg.norm(MU - UM)
    print("\n[2] U = diag(1,-1,1,-1):\n", U)
    print("\nM U =\n", MU)
    print("\nU M =\n", UM)
    print("\nCommute? ||MU - UM||_F =", frob, "(zero means commute)")

    # 3) Centralizer P M = M P
    sol, free_params, cent_basis, rank_poly = centralizer_symbolic(M)
    print("\n[3] Centralizer P M = M P")
    print("Free parameters:", [str(s) for s in free_params])
    print("Rank of {I, M, M^2, M^3} =", rank_poly)
    print("\nOne parametric solution P (entries as linear expressions in free params):\n", sp.simplify(sol))
    print("\nA basis of the centralizer (toggle each free param to 1):")
    for idx, B in enumerate(cent_basis, 1):
        print(f"\n-- Basis #{idx} --\n{B}")

    # 4) U eigensystem (symbolic)
    Us = sp.diag(1, -1, 1, -1)
    print("\n[4] Eigensystem of U (symbolic):")
    for val, mult, vecs in Us.eigenvects():
        print(f"  eigenvalue {val} (mult {mult}); sample vec {vecs[0]}")

    # 5) Pseudo-Hermitian H
    r,w,z,v,x,y = 1,2,3,4,5,6
    H = build_symbolic_H(r,w,z,v,x,y)
    ok1, ok2 = check_pseudo_hermiticity(H, Us)
    print("\n[5] Pseudo-Hermiticity checks for H:")
    print("  H† U = U H  ? ->", ok1)
    print("  H† = U H U  ? ->", ok2)

    # 6) 2x2 toy
    A2, U2, comm = toy_2x2_noncommutation()
    print("\n[6] 2x2 toy example:")
    print("A2 =\n", A2)
    print("U2 =\n", U2)
    print("Commutator A2*U2 - U2*A2 =\n", comm)

if __name__ == "__main__":
    main()
