import numpy as np
import nuclearconstants as nc

# ---------------------------------- #
# --- CROSS SECTION CALCULATIONS --- #
# ---------------------------------- #

def betaFromE(E, m):
    # Calculate beta = v/c from relativistic total energy
    E0 = m * nc.c**2
    return np.sqrt(1-(E0/E)**2)

def cs_coulomb(Z, E, theta):
    # Scattering crossection for pointlike nucleus with coulomb field 
    # a.k.a. Rutherford crossection
    beta = betaFromE(E, nc.me)
    return ((Z * nc.alpha * nc.hbar * nc.c) /
           (2 * beta**2 * E * np.sin(theta/2)**2))**2

def cs_mott(Z, E, theta):
    # Mott model for scattering cross section,
    # relativistic correction to cs_coulomb
    beta = betaFromE(E, nc.me)
    correction = 1 - beta**2 * np.sin(theta/2)**2
    return cs_coulomb(Z, E, theta) * correction


def formFactor(rho_Ch, q2, Z):
    # Form factor of nuclear charge density rho_Ch(r)
    # rho_Ch(r) must be close to zero for r > 8e-15   

    # Sample r
    N = 1000
    r = np.linspace(0, 8, N) * nc.fm

    # Calculate Fourier integral
    result = np.zeros_like(q2)
    for i, q2_val in enumerate(q2):
        q = np.sqrt(q2_val)
        integrand = r * rho_Ch(r) * np.sin(q * r / nc.hbar)
        integral = np.trapz(integrand, x=r)
        fact = 4 * np.pi * nc.hbar/ (Z * nc.e * q)
        result[i] = fact * integral

    return result

def theo_cs(rho_Ch, Z, E, theta):
    # Theoretical crossection of nuclear charge density rho_Ch(r)
    # Product of form factor and Mott crossection

    p2 = (E/nc.c)**2 - (nc.me*nc.c)**2
    q2 = 4 * p2 * np.sin(theta/2)**2
    F2 = formFactor(rho_Ch, q2, Z)**2

    return F2 * cs_mott(Z, E, theta)