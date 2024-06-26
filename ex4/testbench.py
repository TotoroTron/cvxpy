import numpy as np
import admm_lasso as al
import pandas as pd
import time

class Testbench():
    def __init__(self, self, dims, methods, validation_method):
        self._methods = [ validation_method ]
        self._methods.extend(methods)
        self._dims = ...
        self._report = []
        self._dataframe = pd.DataFrame()

    def _verify(self, result, expected):
        return int(not(np.allclose(result, expected, rtol=1e-05, atol=1e-08)))

    def get_report(self):
        return self._report.copy()

    def get_dataframe(self):
        return self._dataframe.copy()
