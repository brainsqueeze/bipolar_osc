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

        self.mu_0 = 0.45e5
        self.R = 10.
        self.deltaR = 0.25
        self.RStar = 40.

        self.u_start = (1. - (self.R / self.RStar) ** 2) ** 0.5
        self.u_stop = 1.
        self.N = 20

        self.E_start = -70.
        self.E_stop = 70.
        self.delta_e = 0.2

        self.Spectra = self.spectra()

        self.x, self.u_prime, self.nu_e_dist, self.nu_mu_dist = np.array(list(zip(*self.Spectra)), dtype=np.float32)
        # self.u_prime = np.unique(self.u_prime)

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

        a = [(p, u, -self.f_nu_e_bar(-p), self.f_nu_mu(-p))
             for p, u in itertools.product(
                np.linspace(start=self.E_start, stop=0.2, num=int((0 - self.E_start) / self.delta_e)),
                np.linspace(start=self.u_start, stop=self.u_stop, num=self.N)
            )
        ]

        a.extend(
            [
                (p, u, self.f_nu_e(p), self.f_nu_mu(p))
                for p, u in itertools.product(
                    np.linspace(start=0.2, stop=self.E_stop, num=int(self.E_stop / self.delta_e)),
                    np.linspace(start=self.u_start, stop=self.u_stop, num=self.N)
                )
            ]
        )

        a.extend(
            [(0, u, self.f_nu_e(0), self.f_nu_mu(0)) for u in np.linspace(self.u_start, self.u_stop, self.N)]
        )

        return sorted(a)

    def mu(self, r):
        """
        Background neutrino density spectrum as a function of outward radius in the SN
        :param r: radius
        :return: Background neutrino density (float)
        """
        return (4 / 3) * self.mu_0 * ((self.R / r) ** 3)


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
        b3 = s13 ** 2 - s23 ** 2 * c13 ** 2 + \
            a * (s12 ** 2 * c13 ** 2 - (c12 * c23 - s12 * s13 * s23 * c_cp) ** 2 - (s12 * s13 * s23 * s_cp) ** 2)

        # 8th component of the vacuum B-vec
        b8 = 3 ** 0.5 * (
                s13 ** 2 + s23 ** 2 * c13 ** 2 +
                a * (s12 ** 2 * c13 ** 2 + (c12 * c23 - s12 * s13 * s23 * c_cp)**2 + (s12 * s13 * s23 * s_cp) ** 2) -
                (2 / 3) * (1 + a)
        )
        return b3, b8

    def sine_func(self, energy_momentum, x_momentum, cosine, u_prime, n):
        """
        Computes sine function which appears inside of the main evolution integral.  All coefficients are computed from
        various tensor contractions with the SU(3) structure tensor.
        :param energy_momentum: un-integrated neutrino energy (float)
        :param x_momentum: background neutrino energy to be integrated (float)
        :param cosine: un-integrate propagation cosine of zenith angle
        :param u_prime: cosine of zenith angle of background neutrino(s)
        :param n: integer which evolves the solution outward, radially in discrete steps
        :return: sine function expansion of vacuum oscillation terms (numpy array)
        """

        if n == 1:
            return np.zeros_like(cosine)

        # if any([n == 1, energy_momentum == 0]):
        #     return 0 * x_momentum * u_prime

        ratio = (1 / (energy_momentum * cosine) - 1 / (x_momentum * u_prime))
        ratio[energy_momentum == 0] = 0

        d = (1 / cosine - u_prime) * (
            0.05119 * np.sin(0.3793 * (n - 1) * self.deltaR * ratio) +
            0.12250 * np.sin(11.895 * (n - 1) * self.deltaR * ratio) +
            0.05410 * np.sin(12.274 * (n - 1) * self.deltaR * ratio)
        )

        # NaNs can/should be set to 0 in this calculation due to the flux
        d[np.isnan(d)] = 0
        return d

    def euler(self, cosine, n, b_orthogonal, log_lam, integral):
        """
        Solution of one iteration of the Euler method for solving the differential equation of motion.
        :param cosine: un-integrate propagation cosine of zenith angle
        :param n: integer which evolves the solution outward, radially in discrete steps
        :param b_orthogonal: orthogonal component of the vacuum oscillation "magnetic field" vector in SU(3) space
        :param log_lam: logarithm of lambda function which controls collective oscillations
        :param integral: results of Riemannian integration at each step
        :return: solution to EOMs at each radius and angle
        """

        if cosine < np.power(1. - np.power((self.R / (self.RStar + (n - 1) * self.deltaR)), 2), 0.5):
            return 0

        return log_lam - self.deltaR * self.mu(self.RStar + (n-1) * self.deltaR) / b_orthogonal * integral

    def vectorized_make_tables(self, previous_solution, n_step):
        log_lambda = previous_solution[:, -1]
        lambda_func = np.exp(log_lambda)

        spectra = lambda_func * self.sine_func(
            energy_momentum=previous_solution[:, 0],
            x_momentum=self.x,
            cosine=previous_solution[:, 1],
            u_prime=self.u_prime,
            n=n_step
        )
        spectra = spectra[:, None]
        spectra = (self.nu_e_dist - self.nu_mu_dist) * spectra

        integrand_array = spectra.reshape(spectra.shape[0] // self.N, self.N, spectra.shape[1])
        # integrand_array = spectra.reshape(spectra.shape[0], spectra.shape[1] // self.N, self.N)

        # # Casts values into a matrix with u_prime = cosine indexing the columns, and the energy indexing the rows
        # # (ie. integrand_array[0, 19] corresponds to energy -70 MeV and u_prime = cosine = 1
        # integrand_array = integrand_array[:, 2].reshape(-1, self.N)

        # Performs first integral over cosine by performing the trapezoidal algorithm over each row of the matrix.
        # cos_int = np.trapz(integrand_array, dx=np.diff(self.u_prime)[0], axis=1)
        cos_int = np.trapz(integrand_array, dx=np.diff(np.unique(self.u_prime))[0], axis=1)

        assert isinstance(cos_int, np.ndarray)
        cos_int[np.isnan(cos_int)] = 0
        return np.trapz(cos_int, dx=0.2, axis=0)

    def make_tables(self, energy_momentum, cosine, n, log_lambda):
        """
        Creates array which will be integrated.
        :param energy_momentum: un-integrated neutrino energy (float)
        :param cosine: un-integrate propagation cosine of zenith angle
        :param n: integer which evolves the solution outward, radially in discrete steps
        :param log_lambda: logarithm of the lambda function from the previous solution
        :return: Integrated function (numpy array)
        """

        log_lam = log_lambda[:, -1]
        lambda_func = np.exp(log_lam)

        # Sets numpy arrays for the integrand
        spectra = (self.nu_e_dist - self.nu_mu_dist) * lambda_func * self.sine_func(
            energy_momentum,
            self.x,
            cosine,
            self.u_prime,
            n
        )
        # integrand_array = np.hstack((self.x[:, None], self.u_prime[:, None], spectra[:, None]))

        # Casts values into a matrix with u_prime = cosine indexing the columns, and the energy indexing the rows
        # (ie. integrand_array[0, 19] corresponds to energy -70 MeV and u_prime = cosine = 1
        # integrand_array = integrand_array[:, 2].reshape(-1, self.N)
        integrand_array = spectra.reshape(-1, self.N)

        # Performs first integral over cosine by performing the trapezoidal algorithm over each row of the matrix.
        cos_int = np.trapz(integrand_array, dx=np.diff(self.u_prime)[0], axis=1)

        assert isinstance(cos_int, np.ndarray)
        cos_int[np.isnan(cos_int)] = 0
        return np.trapz(cos_int, dx=0.2)
