import numpy as np
import lasso as ls 
import pandas as pd
import cvxpy as cvx
import time

class Testbench():
    def __init__(self, params, methods, validation_method):
        self._methods = [ validation_method ]
        self._methods.extend(methods)
        self._params = params
        self._df = pd.DataFrame()

    def _verify(self, result, expectation):
        atol = 1.0 # absolute tolerance
        rtol = 0.1 # relative tolerance
        # result = ( (pstar, xstar), (pstop, xstop), (pfinal, xfinal))
        # expectation = ...
        # just check if the pstars and xstars are close
        failure = int( 0) # failure = 0 means it passed
        failure += int( 1) * int(not(np.allclose( result[0][0], expectation[0][0], rtol, atol )))
        failure += int( 2) * int(not(np.allclose( result[0][1], expectation[0][1], rtol, atol )))
        # failure += int( 4) * int(not(np.allclose( result[1][0], expectation[1][0], rtol, atol)))
        # failure += int( 8) * int(not(np.allclose( result[1][1], expectation[1][1], rtol, atol)))
        # failure += int(16) * int(not(np.allclose( result[2][0], expectation[2][0], rtol, atol)))
        # failure += int(32) * int(not(np.allclose( result[2][1], expectation[2][1], rtol, atol)))
        return failure

    def get_dataframe(self):
        return self._df.copy()

    def test_all(self):
        
        # report = [entry, entry, ..., entry]
        report = []
       
        # BEGIN FOR_PARAMS
        for param in self._params: 
            """
            Initialize variables:
            Properties of Input Matrix A:
                M: height/rows
                N: width/cols
                K: condition_number
                S: sparsity
            Use Scipy to generate matrices with special spectral properties
            """
            M, N, K, S, rho, gamma = param
            np.random.seed(1)
            A = np.random.randn(M, N) # Input data matrix 
            x = cvx.Variable((N, 1)) # Coefficient vector
            b = np.random.rand(M, 1) # Response vector

            # Validation Run
            exp_pstar = None 
            exp_pstop = None
            exp_pfinal = None

            exp_xstar = None
            exp_xstop = None
            exp_xfinal = None

            # Dataframe entries:
            entry = list(param) 

            # BEGIN FOR_METHODS
            for idx, method in enumerate(self._methods):
                inputs = (A, x.copy(), b, rho, gamma)
                instance = method(inputs)
                start_time = time.time()
                instance.solve()
                elapsed_time = time.time() - start_time

                star_point = instance.get_star_point()
                stop_point =  instance.get_stop_point()
                final_point = instance.get_final_point()
                result = (star_point, stop_point, final_point)
                #_point = ( p=f(x), x )

                if idx == 0: # First method as validation run
                    exp_star_point = instance.get_star_point()
                    exp_stop_point = instance.get_stop_point()
                    exp_final_point = instance.get_final_point()
                    expectation = (exp_star_point, exp_stop_point, exp_final_point)

                entry.append( self._verify( result, expectation ) )
                entry.append( elapsed_time )
                entry.append( star_point[0] )
                entry.append( stop_point[0] )
                entry.append( final_point[0] )
            # END FOR_METHODS

            report.append(entry)
        # END ITERATE PARAMS
        
        column_names = [ "M", "N", "K", "S", "rho", "gamma" ]
        for method in self._methods:
            method_name = method.__name__
            column_names.append(f"{method_name}_fail")
            column_names.append(f"{method_name}_time")
            column_names.append(f"{method_name}_pstar")
            column_names.append(f"{method_name}_pstop")
            column_names.append(f"{method_name}_pfinal")
        self._df = pd.concat( [self._df, pd.DataFrame(report)] )
        self._df.columns = column_names
    # END TEST_ALL
        
        



