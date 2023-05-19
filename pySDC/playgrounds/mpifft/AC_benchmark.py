import sys
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import (
    allencahn_imex_timeforcing,
)
from pySDC.projects.AllenCahn_Bayreuth.AllenCahn_dump import dump
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft


def run_simulation(name=""):
    """
    A simple test program to do PFASST runs for the AC equation
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank / int(sys.argv[1]))
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    # space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank % int(sys.argv[1]))
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    # time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()

    # print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank,
    #                                                                     space_size, time_rank, time_size))

    # initialize level parameters
    level_params = dict()
    level_params["restol"] = 1e-08
    level_params["dt"] = 1e-03
    level_params["nsweeps"] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params["quad_type"] = "RADAU-RIGHT"
    sweeper_params["num_nodes"] = [3]
    sweeper_params["QI"] = [
        "LU"
    ]  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params["initial_guess"] = "zero"

    # initialize problem parameters
    problem_params = dict()
    problem_params["L"] = 1.0
    problem_params["nvars"] = [(128, 128), (32, 32)]
    problem_params["eps"] = [0.04]
    problem_params["dw"] = [-23.6]
    problem_params["radius"] = 0.25
    problem_params["comm"] = space_comm
    problem_params["name"] = name
    problem_params["init_type"] = "circle"
    problem_params["spectral"] = False

    # initialize step parameters
    step_params = dict()
    step_params["maxiter"] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params["logger_level"] = (
        20 if space_rank == 0 else 99
    )  # set level depending on rank
    controller_params["hook_class"] = dump

    # fill description dictionary for easy step instantiation
    description = dict()
    # description['problem_class'] = allencahn_imex
    description["problem_class"] = allencahn_imex_timeforcing
    description["problem_params"] = problem_params  # pass problem parameters
    description["sweeper_class"] = imex_1st_order
    description["sweeper_params"] = sweeper_params  # pass sweeper parameters
    description["level_params"] = level_params  # pass level parameters
    description["step_params"] = step_params  # pass step parameters
    description["space_transfer_class"] = fft_to_fft

    # set time parameters
    t0 = 0.0
    Tend = 32 * 0.001

    # instantiate controller
    controller = controller_MPI(
        controller_params=controller_params, description=description, comm=time_comm
    )

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    if space_rank == 0:
        # filter statistics by type (number of iterations)
        iter_counts = get_sorted(stats, type="niter", sortby="time")

        print()

        niters = np.array([item[1] for item in iter_counts])
        out = f"Mean number of iterations on rank {time_rank}: {np.mean(niters):.4f}"
        print(out)

        timing = get_sorted(stats, type="timing_setup", sortby="time")
        out = f"Setup time on rank {time_rank}: {timing[0][1]:.4f} sec."
        print(out)

        timing = get_sorted(stats, type="timing_run", sortby="time")
        out = f"Time to solution on rank {time_rank}: {timing[0][1]:.4f} sec."
        print(out)

        print()

        # convert filtered statistics to list of computed radii, sorted by time
        computed_radii = get_sorted(stats, type="computed_radius", sortby="time")
        exact_radii = get_sorted(stats, type="exact_radius", sortby="time")

        # print radii and error over time
        for cr, er in zip(computed_radii, exact_radii):
            if er[1] > 0:
                err = abs(cr[1] - er[1]) / er[1]
            else:
                err = 1.0
            out = (
                f"Computed/exact/error radius for time {cr[0]:6.4f}: "
                f"{cr[1]:6.4f} / {er[1]:6.4f} / {err:6.4e}"
            )
            print(out)


if __name__ == "__main__":
    name = "AC-test-constforce"
    run_simulation(name=name)
