import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# https://www.cvxpy.org/examples/machine_learning/lasso_regression.html

def loss_fn(X, Y, beta):
    return cp.norm2(X @ beta - Y)**2
    # return cp.sum_of_squares(X @ beta - Y)

def regularizer(beta):
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    # f(x) = ||Ax - B||_2^2 + lambd * ||x||_1
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value

def generate_data(m=100, n=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star

def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")
    plt.savefig("mse_plot.png")

def main():
    m = 100 # data height
    n = 20 # data width
    sigma = 5
    density = 0.2

    # Initialize synthetic data
    X, Y, _ = generate_data(m, n, sigma)
    X_train = X[:50, :]
    Y_train = Y[:50]
    X_test = X[50:, :]
    Y_test = Y[50:]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)
    problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

    lambd_values = np.logspace(-2, 3, 50) # 10e-2 to 10e3, 50 values
    train_errors = []
    test_errors = []
    beta_values = []
    for v in lambd_values:
        lambd.value = v
        problem.solve()
        train_errors.append(mse(X_train, Y_train, beta))
        test_errors.append(mse(X_test, Y_test, beta))
        beta_values.append(beta.value)
    
    print("\nP-STAR: ", problem.value)
    print("\nX-STAR: ", beta.value)

    plot_train_test_errors(train_errors, test_errors, lambd_values)


if __name__ == "__main__":
    main()
