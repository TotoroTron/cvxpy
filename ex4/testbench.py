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
        ...

    def get_dataframe(self):
        return self._df.copy()

    def test_all(self):
        
        # report = [entry, entry, ..., entry]
        # entry = [M, N, K, S, fail1, time1, fail2, time2, etc ...]
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
            # entry = M, N, K, S, fail1, time1, fail2, time2, etc ...] 
            # entry = [M, N, K, S, rho, gamma]
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

                if idx == 0: # First method as validation run
                    exp_star_point = instance.get_star_point()
                    exp_stop_point = instance.get_stop_point()
                    exp_final_point = instance.get_final_point()
                    expectation = (exp_star_point, exp_stop_point, exp_final_point)

                entry.append( self._verify( result, expectation ) )
                entry.append( elapsed_time )
            # END FOR_METHODS

            report.append(entry)
        # END ITERATE PARAMS
        print(report) 
        self._df = pd.concat([ self._df, pd.DataFrame(report) ])
    # END TEST_ALL
        
        




