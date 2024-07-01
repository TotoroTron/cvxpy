import lasso as ls
import testbench as tb

def main():
    params = [ (8, 6, 1.0, 1.0, 1.0, 1.0), (4, 5, 1.0, 1.0, 1.0, 1.0)]
    methods = [ ls.CVXPY_SOLVE ]
    validation_method = ls.CVXPY_SOLVE 
    testbench = tb.Testbench(params, methods, validation_method)
    testbench.test_all()
    df = testbench.get_dataframe()
    print(df.to_string())

if __name__ == "__main__":
    main()
