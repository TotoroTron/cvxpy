import lasso as ls
import testbench as tb

def main():
    K = 1.0
    S = 1.0
    rho = 1.0
    gamma = 1.0

    params = [ (80, 60, K, S, rho, gamma), (250, 100, K, S, rho, gamma) ]

    methods = [ 
        {'method': ls.CVXPY_ADMM_PROX_POOL, 'args': (), 'kwargs': {}}
    ]

    validation_method = {
        'method': ls.CVXPY_CLARABEL, 'args': (), 'kwargs': {'max_iter': 1000}
    }
    # https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/solver.rs

    testbench = tb.Testbench(params, methods, validation_method)
    testbench.test_all()
    df = testbench.get_dataframe()
    print(df.to_string())
    df.to_csv('dataframe.csv')

if __name__ == "__main__":
    main()
