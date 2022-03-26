import numpy as np
import physicsconstants as pc
from crossections import *


def rho_Ch(r, X): 
    (rho0, a, b) = X
    return rho0 / (1 + np.exp((r-a)/b))

def formFactor(q2, X, Z):
    q = np.sqrt(q2)

    # Sample r
    N = 100
    r = np.linspace(0, 8, N) * 1e-15

    # Calculate Fourier integral
    result = np.zeros_like(q2)
    for i,q_val in enumerate(q):
        integrand = r * rho_Ch(r, X) * np.sin(q_val * r / pc.hbar)
        integral = np.trapz(integrand, r)
        fact = 4 * np.pi / (Z * pc.qe * q_val)
        result[i] = fact * integral

    return result

# Plot sample charge density
rho0 = 0.07
a    = 4e-15
b    = 0.5e-15
X = (rho0, a, b)

#r = np.linspace(0, 8) * 1e-15
#plt.plot(r*1e15, rho_Ch(r,X))
#plt.title("Sample charge density curve")
#plt.ylabel("charge density (e fm^-3)")
#plt.xlabel("radius (fm)")
#plt.show()

deg = (2 * np.pi / 360) # One degree in radians
theta = np.linspace(30, 130) * deg

Z = 40
E = 250e6 * pc.eV
p2 = (E/pc.c)**2 - (pc.me*pc.c)**2
q2 = 4 * p2 * np.sin(theta/2)**2

F2 = formFactor(q2, X, Z)**2

fig = plt.figure()
plt.plot(theta / deg, F2 * cs_mott(Z, E, theta) / (1e-3 * pc.barn))
plt.yscale('log')
plt.ylabel("Mott scattering crossection (mb/sr)")
plt.xlabel("angle (deg)")
plt.show()
