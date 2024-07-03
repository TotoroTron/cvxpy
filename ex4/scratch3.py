
from multiprocessing import Pool
import numpy as np
from cvxpy import Minimize, Problem, Variable, norm, sum_squares, SCS 
# Problem data.
m = 100
n = 75
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
gamma = 0.1

NUM_PROCS = 4

def prox(args):
    f, v, x = args
    problem = Problem(Minimize(f + (rho / 2) * sum_squares(x - v)))
    problem.solve()
    return x.value

# Setup problem.
rho = 1.0
x = Variable((n,1))
funcs = [sum_squares(A @ x - b), gamma * norm(x, 1)]
ui = [np.zeros((n, 1)) for _ in funcs]
xbar = np.zeros((n, 1))
pool = Pool(NUM_PROCS)

list_loss = []

# ADMM loop.
for i in range(50):
    prox_args = [xbar - u for u in ui]
    xi = pool.map(prox, [(func, prox_arg, x) for func, prox_arg in zip(funcs, prox_args)])
    xbar = sum(xi) / len(xi)
    ui = [u + x_ - xbar for x_, u in zip(xi, ui)]
    list_loss.append( (sum_squares(A @ xbar - b) + gamma * norm(xbar, 1)).value )


# Compare ADMM with standard solver.
prob = Problem(Minimize(sum(funcs)))
result = prob.solve()

for idx, loss in enumerate(list_loss):
    print(f"idx: {idx}, loss: {loss}")


print("ADMM best", list_loss[-1])
print("ECOS best", result)

data, chain, inverse_data = prob.get_problem_data(SCS)

print("DATA:")
print(data)
print("CHAIN:")
print(chain)
print("INVERSE DATA:")
print(inverse_data)

pool.close()
pool.join()
