import cvxpy as cp
import numpy as np


def main():

    # Problem data.
    m = 30
    n = 20
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Construct the problem.
    x1 = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x1 - b))
    constraints = [0 <= x1, x1 <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print("\nP-STAR: ", prob.value)
    print("\nX-STAR: ", x1.value)
    print("\nRESIDUAL NORM: ", cp.norm(A @ x1 - b, p=2).value)

    print("\nUsing alternate expression for matrix-vector multiplication: \n")

    x2 = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(np.matmul(A, x2) - b))
    # A = (30, 20), x2 = (20)
    # A @ x2 = (30, 1)
    constraints = [0 <= x2, x2 <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    print("\nP-STAR: ", prob.value)
    print("\nX-STAR: ", x2.value)
    print("\nRESIDUAL NORM: ", cp.norm(np.matmul(A, x2) - b, p=2).value)

if __name__ == "__main__":
    main()