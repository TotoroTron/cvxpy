import cvxpy as cp
import numpy


def main():

    # Problem data.
    m = 30
    n = 20
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)

    # Construct the problem.
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    print(x.value)
    # The optimal Lagrange multiplier for a constraint
    # is stored in constraint.dual_value.
    print(constraints[0].dual_value)

if __name__ == "__main__":
    main()