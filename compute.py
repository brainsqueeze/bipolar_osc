from math import *
import numpy as np
from scrub import *
import time
import itertools
import json


R = 10.
RStar = 40.
N = 20.
start = np.power(1. - np.power((R/RStar), 2), 0.5)
stop = 1.
deltaR = 0.25

Dist = spectra() # reads in spectra lists

###########################
#  Main computation code  #
###########################

def BIH():
    # mixing angles and CPV phase
    c12 = np.cos(33.2*np.pi/180)
    c23 = np.cos(40.*np.pi/180)
    c13 = np.cos(8.6*np.pi/180)

    s12 = np.sin(33.2*np.pi/180)
    s23 = np.sin(40.*np.pi/180)
    s13 = np.sin(8.6*np.pi/180)

    cCP = np.cos(300*np.pi/180)
    sCP = np.sin(300*np.pi/180)

    # mass splitting ratio
    a = 7.5*np.power(10., -5)/( 7.5*np.power(10., -5) - 2.43*np.power(10., -3) )

    # 3rd component of the vacuum B-vec
    B3 = np.power(s13,2) - np.power(s23,2)*np.power(c13,2) + a*( np.power(s12,2)*np.power(c13,2) - np.power(c12*c23 - s12*s13*s23*cCP,2) - np.power(s12*s13*s23*sCP,2) )
    
    # 8th component of the vacuum B-vec
    B8 = np.power(3, 0.5)*( np.power(s13,2) + np.power(s23,2)*np.power(c13,2) + a*( np.power(s12,2)*np.power(c13,2) + np.power(c12*c23 - s12*s13*s23*cCP,2) + np.power(s12*s13*s23*sCP,2) ) - (2./3.)*(1 + a) )

    return (B3, B8)



def MakeTables(p, u, n, LogLambda):
    start_time = time.time()
    integ = []
    preInt = []
    
    energy = []
    cosine = []
   

    for x, U, NuEdist, NuMudist in Dist: # x is energy, U is cosine
        cosine.append(U)
        # the coefficients are from the SU(3) symmetry which was computed in a Mathematica notebook based on the derived tensor constractions
            
        preInt.append((NuEdist - NuMudist)*np.exp(LogLambda)*SineFunc(p, x, u, U, n))

        if U == 1.:
            integ.append(np.trapz(preInt, cosine))
            energy.append(x)
            del cosine[:]
            del preInt[:]
                            
                    
    #print '---%s seconds---' % (time.time() - start_time)
    return np.trapz(integ, energy)

    

def integrate():
    Func = []
    
    B = BIH()[0] + np.power(3, -0.5)*BIH()[1]

    # initial LogLambda file, re-defined in the loop below
    LogLambda = [ (p, u, 0) for p in np.arange(-70., 70., 0.2) for u in np.linspace(start, stop, N)]

    for n in range(1, 801):
        
        f = open('Lambda_files/LogLambda_%d_deltaR.txt' % n, 'wb')
        for p, u, loglam in LogLambda:
            if u < np.power(1. - np.power((R/(RStar + (n-1)*deltaR)), 2), 0.5):
                Func.extend([ (p, u, 0) ])
            else:
                Func.extend([ (p, u, loglam - deltaR*Mu(RStar + (n-1)*deltaR)/(B)*MakeTables(p, u, n, loglam) ) ])
                            
                
        del LogLambda[:]
        LogLambda = Func[:]
        del Func[:]

        json.dump(LogLambda, f)
        f.close()

            

def main():
    integrate()
            
                                     
    
if __name__ == '__main__':
    main()
