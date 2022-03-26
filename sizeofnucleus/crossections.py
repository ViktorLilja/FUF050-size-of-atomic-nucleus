import numpy as np
import physicsconstants as pc
import matplotlib.pyplot as plt

# Calculate speed from relativistic total energy
def betaFromE(E, m):
    E0 = m * pc.c**2
    return np.sqrt(1-(E0/E)**2)

# Scattering crossection for pointlike nucleus with coulomb field 
def cs_coulomb(Z, E, theta):
    beta = betaFromE(E, pc.me)
    return ((Z * pc.alpha * pc.hbar * pc.c) /
           (2 * beta**2 * E * np.sin(theta/2)**2))**2

# Mott model for scattering cross section
def cs_mott(Z, E, theta):
    beta = betaFromE(E, pc.me)
    correction = 1 - beta**2 * np.sin(theta/2)**2
    return cs_coulomb(Z, E, theta) * correction

if __name__ == "__main__":

    deg = (2 * np.pi / 360) # One degree in radians
    theta = np.linspace(30, 130) * deg
    Z = 40
    E = 250e6 * pc.eV

    fig = plt.figure()
    plt.plot(theta / deg, cs_mott(Z, E, theta)/(1e-3 * pc.barn))
    plt.yscale('log')
    plt.ylabel("Mott scattering crossection (mb/sr)")
    plt.xlabel("angle (deg)")
    plt.show()