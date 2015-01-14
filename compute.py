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

    x, U, NuEdist, NuMudist = np.array( zip(*Dist) )
    Lambda = np.exp(zip(*LogLambda)[2])


    Integrand = zip( zip(*Dist)[0], zip(*Dist)[1], (NuEdist - NuMudist)*Lambda*SineFunc(p, x, u, U, n) )

    FirstInt = [(zip(*Integrand[x - 20: x])[0][-1], np.trapz(zip(*Integrand[x - 20: x])[2], zip(*Integrand[x - 20: x])[1]) ) for x in range(20, len(Integrand)+20, 20)]


    #print '---%s seconds---' % (time.time() - start_time)
    return np.trapz(zip(*FirstInt)[1], zip(*FirstInt)[0])
   
    

def integrate():
    start_time = time.time()
    Func = []
    
    B = BIH()[0] + np.power(3, -0.5)*BIH()[1]

    # initialize LogLambda file
    LogLambda = [ (round(p, 1), u, 0) for p in np.arange(-70., 70. + 0.2, 0.2) for u in np.linspace(start, stop, N)]
   
    for n in range(1, 2):
        
        f = open('Lambda_files/LogLambda_%d_deltaR.txt' % n, 'wb')

        Func = [(p, u, Euler(u, n, B, loglam, MakeTables(p, u, n, LogLambda)) ) for p, u, loglam in LogLambda ]
                            
                
        del LogLambda[:]
        LogLambda = Func[:]
        del Func[:]

        json.dump(LogLambda, f)
        f.close()
        print '---%s seconds---' % (time.time() - start_time)

                    

def main():
    integrate()
            
                                     
    
if __name__ == '__main__':
    main()
