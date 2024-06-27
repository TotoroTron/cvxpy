import cvxpy as cvx
import numpy as np
import pandas as pd
import abc as ABC


class ADMMLASSO(ABC):
    def __init__(self, A, x, b):
        self._A = A # Pass in by reference
        self._x = x # Pass in by copy
        self._b = b # Pass in by reference

    def ADMM():
        # Perform some calculation using A x and b and alter x.
        ...

    def LASSO():
        ...

    



