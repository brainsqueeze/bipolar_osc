from math import *
import numpy as np
import matplotlib.pyplot as plt
import sys

L = 1.5*10**51
FluxTot = L*(10**-1 + 15**-1 + 4*20**-1)
Mu0 = 0.45*10**5
R = 40
deltaR = 0.25
deltaU = sqrt(1 - (10*R**-1)**2)/20
deltaP = 0.2


def fnuE (x):
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
    T = 2.1
    Eta = 3.9
    
    return (FluxE/FluxTot)*np.power(x/EeAvg, 2.)*np.power(1 + np.exp(x/T - Eta), -1)

def main():
    x = np.linspace(0., 50.)
    line, = plt.plot(x, fnuE(x), '--', linewidth=2)

    plt.show()


if __name__ == '__main__':
    main()
