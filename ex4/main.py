import lasso as ls
import testbench as tb

def main():
    params = [ (80, 60, 1.0, 1.0, 1.0, 1.0), (40, 50, 1.0, 1.0, 1.0, 1.0)]
    methods = [ ls.CVXPY_ADMM_POOL ]
    validation_method = ls.CVXPY_SOLVE 
    testbench = tb.Testbench(params, methods, validation_method)
    testbench.test_all()
    df = testbench.get_dataframe()
    print(df.to_string())
    df.to_csv('dataframe.csv')

if __name__ == "__main__":
    main()
