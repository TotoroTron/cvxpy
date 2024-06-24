from cvxpy import *
import numpy as np
from multiprocessing import Process, Pipe

# https://www.cvxpy.org/examples/applications/consensus_opt.html

def run_worker(f, pipe):
    xbar = Parameter(n, value=np.zeros(n))
    u = Parameter(n, value=np.zeros(n))
    f += (rho/2)*sum_squares(x - xbar + u)
    prox = Problem(Minimize(f))
    # ADMM loop:
    while True:
        prox.solve()
        pipe.send(x.value)
        xbar.value = pipe.recv()
        u.value += x.value - xbar.value

def main():

    # Number of terms f_i
    N = ...
    # A list of all f_i.
    f_list = ...

   # Setup the workers.
    pipes = []
    procs = []
    for i in range(N):
        local, remote = Pipe()
        pipes += [local]
        procs += [Process(target=run_process, args=(f_list[i], remote))]
        proces[-1].start()

    # ADMM loop
    for i in range(MAX_ITER):
        # Gather and average xi
        xbar = sum(pipe.recv() for pipe in pipes) / N

        # Scatter xbar
        for pipe in pipes:
            pipe.send(xbar)

    [p.terminate() for p in procs]


if __name__ == "__main__":
    main()
