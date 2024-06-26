import numpy as np
import admm_lasso as al
import pandas as pd
import time

class Testbench():
    def __init__(self, self, params, methods, validation_method):
        self._methods = [ validation_method ]
        self._methods.extend(methods)
        self._params = params
        self._dataframe = pd.DataFrame()

    def _verify(self, result, expected):
        return int(not(np.allclose(result, expected, rtol=1e-05, atol=1e-08)))

    def get_dataframe(self):
        return self._dataframe.copy()

    def test_all(self):
        
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
            M, N, K, S = params
            A = np.randn(randn(M, N)) # Input data matrix 
            x = np.randn(randn(N)) # Coefficient vector
            b = ... # Response vector

            # Validation Run
            expectation = None

            # Dataframe entries:
            # entry = M, N, K, S, mthd1_fail, mthd1_time, mthd2_fail, mthd2_time, ... 
            entry = [M, N, K, S]

            for idx, method in enumerate(self._methods):
                instance = method(A, x.copy(), b)
                start_time = time.time()
                instance.run()
                elapsed_time = time.time() - start_time
                result = instance.get_result()

                if idx == 0: # First method as validation run
                    expectation = instance.get_result()

                entry.extend( self._verify( result, expectation ) )
                entry.extend( elapsed_time )

            self._dataframe.loc[len(self._dataframe)] = entry 

            ...
        ...







