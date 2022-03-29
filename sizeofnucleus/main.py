from sysconfig import get_path
import numpy as np
from scipy.optimize import minimize
import physicsconstants as pc
from crossections import *
import pandas as pd
import os

# GLOBAL VARIABLES #
E = 250e6 * pc.eV
A = 40
Z = 20


def rho_Ch(r, X): 
    (rho0, a, b) = X
    return rho0 / (1 + np.exp((r-a)/b))

def formFactor(q2, X, Z):
    (rho0, a, b) = X
    #assert (a < 7*pc.fm) #since we assume that the nucleus is only a few fm
    q = np.sqrt(q2)

    # Sample r
    N = 1000
    r = np.linspace(0, 8, N) * pc.fm

    # Calculate Fourier integral
    result = np.zeros_like(q2)
    for i,q_val in enumerate(q):
        integrand = r * rho_Ch(r, X) * np.sin(q_val * r / pc.hbar)
        integral = np.trapz(integrand, x=r)
        fact = 4 * np.pi * pc.hbar/ (Z * pc.e * q_val)
        result[i] = fact * integral

    return result

def theo_cs(Z, E, X, theta):
    p2 = (E/pc.c)**2 - (pc.me*pc.c)**2
    q2 = 4 * p2 * np.sin(theta/2)**2
    F2 = formFactor(q2, X, Z)**2

    return F2 * cs_mott(Z, E, theta)

def proton_count(X):
    #assert (a < 7*pc.fm) #since we assume that the nucleus is only a few fm
    N = 1000
    r = np.linspace(0, 8, N) * pc.fm

    integrand = r**2 * rho_Ch(r, X)
    integral = np.trapz(integrand, x=r)
    fact = 4 * np.pi / pc.e

    return fact * integral 



#r = np.linspace(0, 8) * 1e-15
#plt.plot(r*1e15, rho_Ch(r,X))
#plt.title("Sample charge density curve")
#plt.ylabel("charge density (e fm^-3)")
#plt.xlabel("radius (fm)")
#plt.show()

rho0 = 0.07 * pc.e * (pc.fm)**(-3)
a    = 4e-15
b    = 0.5e-15
X = (rho0, a, b)

deg = (2 * np.pi / 360) # One degree in radians
theta = np.linspace(30, 130, 500) * deg


# Plot sample charge density
fig = plt.figure()
plt.plot(theta / deg, theo_cs(Z, E, X, theta) / (1e-3 * pc.barn))
plt.yscale('log')
plt.ylabel("Mott scattering crossection (mb/sr)")
plt.xlabel("angle (deg)")
plt.show()






# NUMERICAL OPTIMIZATION #

current_path = os.path.dirname(os.path.abspath(__file__))
CSV_fileName = "experimentalData.csv"
CSV_data     = pd.read_csv(current_path + "\\" + CSV_fileName, sep=',')

header    = CSV_data.columns.values
exp_theta = CSV_data[header[0]] * (np.pi / 180)
exp_cs    = CSV_data[header[1]] * (1e-3 * pc.barn)
exp_error = CSV_data[header[2]] * (1e-3 * pc.barn)

def chi2(X):
    epsilon = 1e-6

    err = np.sum( ( (theo_cs(Z, E, X, exp_theta) - exp_cs) / exp_error )**2 )
    constraint = ( (Z - proton_count(X))/epsilon )**2  # correct number of protons in nucleus
    chi2 = err + constraint

    return chi2


# X_star = argmin( chi2(X)) 

X_star = minimize(chi2, X)
print(X_star)