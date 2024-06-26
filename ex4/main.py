from multiprocessing import Pool

import numpy as np

from cvxpy import Minimize, Problem, Variable, norm, sum_squares

# Problem data.
m = 100
n = 75
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
gamma = 0.1

NUM_PROCS = 4

def prox(args):
    f, v = args
    x = Variable((n, 1))
    f += (rho/2)*sum_squares(x - v)
    Problem(Minimize(f)).solve()
    return x.value

# Setup problem.
rho = 1.0
x = Variable((n, 1))
funcs = [sum_squares(A @ x - b), gamma * norm(x, 1)]
ui = [np.zeros((n, 1)) for func in funcs]
xbar = np.zeros((n, 1))
pool = Pool(NUM_PROCS)
# ADMM loop.
for i in range(50):
    prox_args = [xbar - u for u in ui]
    xi = pool.map(prox, zip(funcs, prox_args))
    xbar = sum(xi)/len(xi)
    ui = [u + x_ - xbar for x_, u in zip(xi, ui)]

# Compare ADMM with standard solver.
prob = Problem(Minimize(sum(funcs)))
result = prob.solve()
print("ADMM best", (sum_squares(np.dot(A, xbar) - b) + gamma * norm(xbar, 1)).value)
print("ECOS best", result)
