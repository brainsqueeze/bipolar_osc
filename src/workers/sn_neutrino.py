import numpy as np
import itertools


class SNThermodynamics(object):

    def __init__(self):
        self.Le = 4.1 * np.power(10., 51.)
        self.Lebar = 4.3 * np.power(10., 51.)
        self.Lmu = 7.9 * np.power(10., 51.)

        self.EeAvg = 9.4
        self.EeBarAvg = 13.
        self.EmuAvg = 15.8

        self.FluxE = self.Le / self.EeAvg
        self.FluxEbar = self.Lebar / self.EeBarAvg
        self.FluxMu = self.Lmu / self.EmuAvg
        self.FluxTot = self.FluxE + self.FluxEbar + 4 * self.FluxMu

        self.TE = 2.1
        self.TEbar = 3.5
        self.TMu = 4.4
        self.EtaE = 3.9
        self.EtaEbar = 2.3
        self.EtaMu = 2.1

        self.mu_0 = 0.45 * np.power(10., 5)
        self.R = 10.
        self.deltaR = 0.25
        self.RStar = 40.

        self.u_start = (1. - (self.R / self.RStar)**2)**0.5
        self.u_stop = 1.
        self.N = 20

        self.E_start = -70.
        self.E_stop = 70.
        self.delta_e = 0.2

        self.Spectra = self.spectra()

    def f_nu_e(self, x):
        """
        Electron neutrino initial SN flux
        :param x: energy (+ for nu, - for anti nu)
        :return: initial energy spectrum (numpy array)
        """
        return (self.FluxE / self.FluxTot) * np.power(x / self.EeAvg, 2.) * \
               np.power(1 + np.exp(x / self.TE - self.EtaE), -1)

    def f_nu_e_bar(self, x):
        """
        Anti-electron neutrino initial SN flux
        :param x: energy (+ for nu, - for anti nu)
        :return: initial energy spectrum (numpy array)
        """
        return (self.FluxEbar / self.FluxTot) * np.power(x / self.EeBarAvg, 2.) * \
               np.power(1 + np.exp(x / self.TEbar - self.EtaEbar), -1)

    def f_nu_mu(self, x):
        """
        Muon / tau neutrino initial SN flux
        :param x: energy (+ for nu, - for anti nu)
        :return: initial energy spectrum (numpy array)
        """
        return (self.FluxMu / self.FluxTot) * np.power(x / self.EmuAvg, 2.) * \
               np.power(1 + np.exp(x / self.TMu - self.EtaMu), -1)

    def spectra(self):
        """
        :return: total initial spectrum - including all flavors - as a function of energy and cosine of zenith (array)
        """
        a = [(round(p, 1), u, -self.f_nu_e_bar(-p), -self.f_nu_mu(-p))
             for p in np.linspace(self.E_start, 0.2, round((0 - self.E_start) / self.delta_e, 0))
             for u in np.linspace(self.u_start, self.u_stop, self.N)]
        a.extend([(round(p, 1), u, self.f_nu_e(p), self.f_nu_mu(p))
                  for p in np.linspace(0.2, self.E_stop, round((self.E_stop - 0) / self.delta_e, 0))
                  for u in np.linspace(self.u_start, self.u_stop, self.N)])
        a.extend([(0, u, self.f_nu_e(0), self.f_nu_mu(p))
                  for u in np.linspace(self.u_start, self.u_stop, self.N)])
        return sorted(a)

    def mu(self, r):
        """
        Background neutrino density spectrum as a function of outward radius in the SN
        :param r: radius
        :return: Background neutrino density (float)
        """
        return (4./3.) * self.mu_0 * np.power((self.R / r), 3)


class Neutrino(SNThermodynamics):

    def __init__(self):
        super(Neutrino, self).__init__()

    @staticmethod
    def b_inverted():
        """
        B-vector in the inverted mass hierarchy.
        :return: 3rd and 8th components of B
        """
        # mixing angles and CPV phase
        c12 = np.cos(33.2 * np.pi/180)
        c23 = np.cos(40. * np.pi/180)
        c13 = np.cos(8.6 * np.pi/180)

        s12 = np.sin(33.2 * np.pi/180)
        s23 = np.sin(40. * np.pi/180)
        s13 = np.sin(8.6 * np.pi/180)

        c_cp = np.cos(300 * np.pi/180)
        s_cp = np.sin(300 * np.pi/180)

        # mass splitting ratio
        a = 7.5 * np.power(10., -5) / (7.5 * np.power(10., -5) - 2.43 * np.power(10., -3))

        # 3rd component of the vacuum B-vec
        b3 = s13**2 - s23**2 * c13**2 + \
             a * (s12**2 * c13**2 - (c12 * c23 - s12 * s13 * s23 * c_cp)**2 - (s12 * s13 * s23 * s_cp)**2)

        # 8th component of the vacuum B-vec
        b8 = 3**0.5 * (s13**2 + s23**2 * c13**2 +
                       a * (s12**2 * c13**2 + (c12 * c23 - s12 * s13 * s23 * c_cp)**2 + (s12 * s13 * s23 * s_cp)**2) -
                       (2. / 3.) * (1 + a))
        return b3, b8

    def sine_func(self, p, x, u, u_prime, n):
        """
        Computes sine function which appears inside of the main evolution integral.  All coefficients are computed from
        various tensor contractions with the SU(3) structure tensor.
        :param p: un-integrated neutrino energy (float)
        :param x: background neutrino energy to be integrated (float)
        :param u: un-integrate propagation cosine of zenith angle
        :param u_prime: cosine of zenith angle of background neutrino(s)
        :param n: integer which evolves the solution outward, radially in discrete steps
        :return: sine function expansion of vacuum oscillation terms (numpy array)
        """
        if any([n == 1, p == 0]):
            return 0 * x * u_prime
        else:
            d = (1/u - u_prime) * (
                0.05119 * np.sin(0.3793 * (n-1) * self.deltaR * (1 / (p * u) - 1 / (x * u_prime))) +
                0.1225 * np.sin(11.895 * (n-1) * self.deltaR * (1 / (p * u) - 1 / (x * u_prime))) +
                0.0541 * np.sin(12.274 * (n-1) * self.deltaR * (1 / (p * u) - 1 / (x * u_prime)))
            )
            nans = np.isnan(d)  # NaNs can/should be set to 0 in this calculation due to the flux
            d[nans] = 0
            return d

    def euler(self, u, n, b_orthogonal, log_lam, integral):
        """
        Solution of one iteration of the Euler method for solving the differential equation of motion.
        :param u: un-integrate propagation cosine of zenith angle
        :param n: integer which evolves the solution outward, radially in discrete steps
        :param b_orthogonal: orthogonal component of the vacuum oscillation "magnetic field" vector in SU(3) space
        :param log_lam: logarithm of lambda function which controls collective oscillations
        :param integral: results of Riemannian integration at each step
        :return: solution to EOMs at each radius and angle
        """
        if u < np.power(1. - np.power((self.R / (self.RStar + (n-1) * self.deltaR)), 2), 0.5):
            return 0
        else:
            return log_lam - self.deltaR * self.mu(self.RStar + (n-1) * self.deltaR) / b_orthogonal * integral

    def make_tables(self, p, u, n, log_lambda):
        """
        Creates array which will be integrated.
        :param p: un-integrated neutrino energy (float)
        :param u: un-integrate propagation cosine of zenith angle
        :param n: integer which evolves the solution outward, radially in discrete steps
        :param log_lambda: logarithm of the lambda function from the previous solution
        :return: Integrated function (numpy array)
        """
        x, u_prime, nu_e_dist, nu_mu_dist = np.array(zip(*self.Spectra))
        _, _, log_lam = zip(*log_lambda)
        lambda_func = np.exp(log_lam)

        # Sets numpy arrays for the integrand
        integrand_array = np.array(zip(x, u_prime,
                                       (nu_e_dist - nu_mu_dist) * lambda_func * self.sine_func(p, x, u, u_prime, n)))

        """
        Casts values into a matrix with u_prime = cosine indexing the columns
        and the energy indexing the rows (ie. integrand_array[0, 19] corresponds
        to energy = -70 MeV and u_prime = cosine = 1).
        """
        integrand_mat = integrand_array[:, 2].reshape((len(integrand_array) / 20, 20))
        """
        Performs first integral over cosine by performing the trapezoidal
        algorithm over each row of the matrix.
        """
        cos_int = np.trapz(integrand_mat, dx=np.diff(u_prime)[0], axis=1)

        cos_int[np.isnan(cos_int)] = 0
        return np.trapz(cos_int, dx=0.2)
