import numpy as np
import h5py

import warnings
import time
import os

from .workers.sn_neutrino import Neutrino

warnings.filterwarnings(action="ignore")
root = os.path.dirname(os.path.abspath(__file__))
nu = Neutrino()
MIXING_ANGLES = nu.b_inverted()
B_VECTOR = MIXING_ANGLES[0] + 3 ** -0.5 * MIXING_ANGLES[1]


def log(step_num, elapsed_time):
    print(f"[INFO] Finished step {step_num} in {elapsed_time} seconds")


def solve_fixed_energy_angle(previous_solution_space, phase_space_position, step_num):
    """
    Solves the differential equation for a fixed cosine (trajectory angle) and neutrino energy
    :param previous_solution_space: solution to previous step of the differential equation
    across all angles and energies, includes phase space values in first 2 columns (ndarray)
    :param phase_space_position: which cosine / neutrino energy slice for which to compute the solution
    (int, row index of previous solution)
    :param step_num: step number in discretized radial distance away from the initial core (int)
    :return: solution (ndarray)
    """

    energy_momentum, cosine, previous_solution = previous_solution_space[phase_space_position]

    inner_integral = nu.make_tables(
        energy_momentum=energy_momentum,
        cosine=cosine,
        n=step_num,
        log_lambda=previous_solution_space
    )

    euler_step_solution = nu.euler(
        cosine=cosine,
        n=step_num,
        b_orthogonal=B_VECTOR,
        log_lam=previous_solution,
        integral=inner_integral
    )
    return euler_step_solution


def solve_step(previous_solution_space, phase_space_position, step_num):
    """
    Solves the differential equation across the full spectrum of trajectory angles and neutrino energies
    :param previous_solution_space: solution to previous step of the differential equation
    across all angles and energies, includes phase space values in first 2 columns (ndarray)
    :param phase_space_position: which cosine / neutrino energy slice for which to compute the solution
    (int, row index of previous solution)
    :param step_num: step number in discretized radial distance away from the initial core (int)
    :return: solution (ndarray)
    """

    euler_solution = solve_fixed_energy_angle(
        previous_solution_space=previous_solution_space,
        phase_space_position=phase_space_position,
        step_num=step_num
    )

    return previous_solution_space[phase_space_position][0], previous_solution_space[phase_space_position][1], euler_solution


def save(solution_space, step_num):
    """
    Saves each solution + phase space to HDF formatted files
    :param solution_space: solution of the differential equation at a fixed step,
    across all angles and energies, includes phase space values in first 2 columns (ndarray)
    :param step_num: step number in discretized radial distance away from the initial core (int)
    """

    h5f = h5py.File(root + f"/data/LogLambda_{step_num}_deltaR.h5", "w")
    h5f.create_dataset("step_solution", data=solution_space, dtype=np.float32)
    h5f.close()


def solve(total_steps=4):
    """
    Solves the differential equation for a fixed cosine (trajectory angle) and neutrino energy
    :param total_steps: total number of steps in discretized radial distance away from the
    initial core for which to solve (int)
    """

    assert isinstance(total_steps, int) and total_steps > 1

    # initialize LogLambda file
    log_lambda = np.array([
        (
            round(energy_momentum, 1),
            cosine,
            0
        ) for energy_momentum in np.arange(nu.E_start, nu.E_stop + nu.delta_e, nu.delta_e)
        for cosine in np.linspace(nu.u_start, nu.u_stop, nu.N)
    ], dtype=np.float32)
    save(solution_space=log_lambda, step_num=0)

    for n in range(1, total_steps):
        start_time = time.time()

        log_lambda = np.array([
            solve_step(previous_solution_space=log_lambda, phase_space_position=row, step_num=n)
            for row in range(log_lambda.shape[0])
        ])
        save(solution_space=log_lambda, step_num=n)
        log(step_num=n, elapsed_time=time.time() - start_time)


if __name__ == '__main__':
    solve(total_steps=10)
