#!/usr/bin/env python3
import sympy as sp
from surfacedispersion import (
    surface_signature_U, build_block_surface_H, is_pseudo_hermitian
)

def main():
    print("=== surfacedispersion: surface model demo (001surface.nb) ===")
    U = surface_signature_U()
    print("\nU = diag(1,-1,1,-1):\n", U)

    # Sample parameters (real)
    r,w,z,v,x,y = 1,2,3,4,5,6
    H = build_block_surface_H(r,w,z,v,x,y)
    print("\nStructured 4x4 H(r,w,z,v,x,y):\n", H)

    ok1, ok2 = is_pseudo_hermitian(H, U)
    print("\nPseudo-Hermiticity checks:")
    print("  H† U = U H  ? ->", ok1)
    print("  H† = U H U  ? ->", ok2)

    # Show that equalities are symbolic identities (zero matrices)
    Hdag = H.conjugate().T
    print("\nH† U - U H:\n", sp.simplify(Hdag*U - U*H))
    print("\nH† - U H U:\n", sp.simplify(Hdag - U*H*U))

if __name__ == "__main__":
    main()
