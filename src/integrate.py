import numpy as np
import time

from workers.sn_neutrino import Neutrino


def solve(obj):
    b = Neutrino.b_inverted()[0] + 3**-0.5 * Neutrino.b_inverted()[1]

    # initialize LogLambda file
    log_lambda = [(round(p, 1), u, 0) for p in np.arange(obj.E_start, obj.E_stop + obj.delta_e, obj.delta_e)
                  for u in np.linspace(obj.u_start, obj.u_stop, obj.N)]
    print len(log_lambda)

    for n in range(1, 4):

        f = open('data/LogLambda_%d_deltaR.txt' % n, 'wb')

        start_time = time.time()
        func = [(p, u, obj.euler(u, n, b, log_lam, obj.make_tables(p, u, n, log_lambda)))
                for p, u, log_lam in log_lambda]

        del log_lambda[:]
        log_lambda = func[:]
        del func[:]

        np.savetxt(f, log_lambda, delimiter=", ")
        f.close()
        print '---%s seconds---' % (time.time() - start_time)


def main():
    nu = Neutrino()  # initializes the class
    solve(obj=nu)


if __name__ == '__main__':
    main()
