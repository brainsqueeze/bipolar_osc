from math import *
import numpy as np

Le = 4.1*np.power(10.,51.)
Lebar = 4.3*np.power(10.,51.)
Lmu = 7.9*np.power(10.,51.)
EeAvg = 9.4
EeBarAvg = 13.
EmuAvg = 15.8
FluxE = Le/EeAvg
FluxEbar = Lebar/EeBarAvg
FluxMu = Lmu/EmuAvg
FluxTot = FluxE + FluxEbar + 4*FluxMu

deltaR = 0.25

####################
#  Initial fluxes  #
####################

def fnuE(x):
    T = 2.1
    Eta = 3.9

    return (FluxE/FluxTot)*np.power(x/EeAvg, 2.)*np.power(1 + np.exp(x/T - Eta), -1)

def fnuEbar(x):
    T = 3.5
    Eta = 2.3

    return (FluxEbar/FluxTot)*np.power(x/EeBarAvg, 2.)*np.power(1 + np.exp(x/T - Eta), -1)

def fnuMu(x):
    T = 4.4
    Eta = 2.1
    
    return (FluxMu/FluxTot)*np.power(x/EmuAvg, 2.)*np.power(1 + np.exp(x/T - Eta), -1)
  

def spectra():
    start, stop, deltaE = -70., 70., 0.2
    Ustart, Ustop, N = np.power(1. - np.power((10./40.), 2), 0.5), 1., 20.

    a = [(round(p,1), u, -fnuEbar(-p), -fnuMu(-p)) for p in np.linspace(start, 0.2, round((0 - start)/deltaE, 0)) for u in np.linspace(Ustart, Ustop, N)]
    a.extend([(round(p,1), u, fnuE(p), fnuMu(p)) for p in np.linspace(0.2, stop, round((stop - 0)/deltaE, 0)) for u in np.linspace(Ustart, Ustop, N)])
    a.extend([(0, u, fnuE(0), fnuMu(p)) for u in np.linspace(Ustart, Ustop, N)])

    return sorted(a)



def Mu(r):
    mu0 = 0.45*np.power(10., 5)
    R = 10.
    return (4./3.)*mu0*np.power((R/r), 3)




def SineFunc(p, x, u, U, n):
    if any([n == 1, p == 0]):
        return 0*x*U
    else:
        D = (1/u - U)*( 0.05119*np.sin( 0.3793*(n-1)*deltaR*(1/(p*u) - 1/(x*U)) ) + 0.1225*np.sin( 11.895*(n-1)*deltaR*(1/(p*u) - 1/(x*U)) ) + 0.0541*np.sin( 12.274*(n-1)*deltaR*(1/(p*u) - 1/(x*U)) ) )

        # NaNs can/should be set to 0 in this calculation due to the flux
        NaNs = np.isnan(D)
        D[NaNs] = 0
        return D



def Euler(u, n, B, loglam, Sum):
    R, RStar = 10., 40.
    
    if u < np.power(1. - np.power((R/(RStar + (n-1)*deltaR)), 2), 0.5):
        return 0
    else:
        return loglam - deltaR*Mu(RStar + (n-1)*deltaR)/(B)*Sum


