import numpy as np
from scipy.optimize import minimize, Bounds
import physicsconstants as pc
from crossections import *
import pandas as pd
import os

# --- IMPORT DATA --- #

# Experimental parameters
E = 250e6 * pc.eV
A = 40
Z = 20

# Get experimental data from file
current_path = os.path.dirname(os.path.abspath(__file__))
CSV_fileName = "data/experimentalData.csv"
CSV_data     = pd.read_csv(current_path + "/" + CSV_fileName, sep=',')

header    = CSV_data.columns.values
exp_theta = np.array(CSV_data[header[0]]) * (np.pi / 180)
exp_cs    = np.array(CSV_data[header[1]]) * (1e-3 * pc.barn)
exp_error = np.array(CSV_data[header[2]]) * (1e-3 * pc.barn)


# --- HELPER FUNCTIONS --- #

def rho_Ch(r, X):
    # Theoretical model for charge distribution.
    # Returns volume charge density at r given parameters X (si units)
    (rho0, a, b) = X
    return rho0 / (1 + np.exp((r-a)/b))

def proton_count(rho_Ch):
    # Calculate charge density for a given charge distribution
    # rho_Ch(r) must be close to zero for r > 8e-15   
    N = 1000
    r = np.linspace(0, 8, N) * pc.fm

    integrand = r**2 * rho_Ch(r)
    integral = np.trapz(integrand, x=r)
    fact = 4 * np.pi / pc.e

    return fact * integral

def rms_radius(rho_Ch):
    # Calculate rms radius for a given charge distribution
    # rho_Ch(r) must be close to zero for r > 8e-15   
    N = 1000
    r = np.linspace(0, 8, N) * pc.fm

    integrand = r**4 * rho_Ch(r)
    integral = np.trapz(integrand, x=r)
    fact = 4 * np.pi / (Z * pc.e)

    return np.sqrt(fact * integral)


# --- NUMERICAL OPTIMIZATION --- #

def X_nuclear2si(X_nuclear):
    # Helper function:
    # Convert X from nuclear units to SI units
    (rho0, a, b) = X_nuclear
    rho0 *= pc.e * pc.fm**-3
    a    *= pc.fm
    b    *= pc.fm
    X_si = (rho0, a, b)
    return X_si

def chi2(X_nuclear):
    # chi2 error funtion to be minimized
    # X_nuclear given in (e/fm3, fm, fm)
    X = X_nuclear2si(X_nuclear)

    # Chi2 for cross section fit to data
    epsilon = 1e-6
    diff = theo_cs(lambda r: rho_Ch(r, X), Z, E, exp_theta) - exp_cs
    err = np.sum((diff / exp_error ) ** 2)

    # Constraint: correct number of protons in nucleus
    constraint = ( (Z - proton_count(lambda r: rho_Ch(r, X)))/epsilon )**2  
    chi2 = err + constraint

    # Print current result (for optimization visualization)
    (rho0, a, b) = X_nuclear
    print("rho0 = %.4f e/fm3, a = %.3f fm, b = %.3f fm, chi2 = %.2E" 
          % (rho0, a, b, chi2))

    return chi2

# Initial guess
rho0  = 0.0743             # [e/fm3]
a0    = 3.77               # [fm]
b0    = 0.538              # [fm]
X0 = (rho0, a0, b0)

# Linear constraint: lb <= X <= ub
lb = np.array([0.06, 3, 0.3]) # rho, a, b
ub = np.array([0.08, 5, 0.7])
bounds = Bounds(lb, ub)

# Optimize using scipy default minimization function
res = minimize(chi2, X0,
               tol=1e-6, 
               options={"maxiter": 1000},
               bounds=bounds)
print(res)
X_star = res.x
X_star_SI = X_nuclear2si(X_star)

# Charge distribution and crossection of optimal fit
rho_Ch_star = lambda r: rho_Ch(r, X_star_SI)


# --- SHOW RESULT --- #

# Print RMS
rms_acc         = 3.4776 * pc.fm           # Accepted rms
rms_exp         = rms_radius(rho_Ch_star)  # Our rms
rms_rel_error   = np.abs((rms_exp - rms_acc) / rms_acc)

print("Total charge %f e" % proton_count(rho_Ch_star))
print("Expected RMS radius %f fm" % (rms_acc / pc.fm))
print("Calculated RMS radius %f fm, relative error %.2f%%" 
       % (rms_radius(rho_Ch_star)/pc.fm, 100*rms_rel_error))


# Plot cross section
deg = (2 * np.pi / 360) # One degree in radians
theta = np.linspace(30, 130, 500) * deg
cs_star = theo_cs(rho_Ch_star, Z, E, theta)
plt.plot(theta / deg, 
         cs_star / (1e-3 * pc.barn),
         "k--")
plt.errorbar(exp_theta / deg, 
             exp_cs / (1e-3 * pc.barn), 
             yerr = exp_error / (1e-3 * pc.barn),
             fmt = "r.")
plt.yscale('log')
plt.ylabel("Mott scattering crossection (mb/sr)")
plt.xlabel("angle (deg)")
plt.grid(True)
plt.show()

# Plot charge distribution
r = np.linspace(0, 8) * pc.fm
plt.plot(r*1e15, 
         rho_Ch_star(r) / (pc.e * pc.fm**-3))
plt.title("Charge distribution of Ca nucleus")
plt.ylabel("charge density (e fm^-3)")
plt.xlabel("radius (fm)")
plt.show()


# --- EXPORT RESULT TO CSV --- #
with open('sizeofnucleus/output/result.csv', 'w') as f:
    # Header:
    f.write("exp_theta (deg),exp_cs (mb/sr),exp_error (mb/sr),fit_theta (deg),fit_cs (mb/sr),r (fm),rho_Ch (e/fm3)\n")
    rows = max(len(exp_theta), len(theta), len(r))
    for i in range(0, rows):
        if (len(exp_theta)      > i) : f.write("%e" % (exp_theta[i] / deg))
        f.write(",")
        if (len(exp_cs)         > i) : f.write("%e" % (exp_cs[i] / (1e-3 * pc.barn)))
        f.write(",")
        if (len(exp_error)      > i) : f.write("%e" % (exp_error[i] / (1e-3 * pc.barn)))
        f.write(",")
        if (len(theta)          > i) : f.write("%e" % (theta[i] / deg))
        f.write(",")
        if (len(cs_star)        > i) : f.write("%e" % (cs_star[i] / (1e-3 * pc.barn)))
        f.write(",")
        if (len(r)              > i) : f.write("%e" % (r[i] / pc.fm))
        f.write(",")
        if (len(rho_Ch_star(r)) > i) : f.write("%e" % (rho_Ch_star(r)[i] / (pc.e * pc.fm**-3)))
        f.write("\n")

