import lasso as ls
import testbench as tb

def main():
    K = 1.0
    S = 1.0
    rho = 1.0
    gamma = 1.0

    params = [ (80, 60, K, S, rho, gamma), (250, 100, K, S, rho, gamma) ]

    methods = [ 
        {
            'method': ls.ADMM_PROX_POOL,
            'kwargs': {'num_procs': 4, 'max_iter': 400},
        },
       # {
       #     'method': ls.ADMM_MPI,
       #     'kwargs': {},
       # },
    ]

    validation_method = { 
        'method': ls.CLARABEL, 
        'kwargs': {'max_iter': 400} 
    }

    # https://clarabel.org/stable/api_settings/
    # https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/solver.rs

    testbench = tb.Testbench(params, methods, validation_method)
    testbench.test_all()
    df = testbench.get_dataframe()
    print(df.to_string())
    df.to_csv('dataframe.csv')

if __name__ == "__main__":
    main()
