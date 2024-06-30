import admm_lasso as al 
import testbench as tb

def main():
    params = (100, 80, 1.0, 1.0)
    methods = [ al.ADMM_POOL ]
    validation_method = [ al.CVXPY_SOLVE ]
    testbench = tb.Testbench(params, methods, validation_method)

if __name__ == "__main__":
    main()
