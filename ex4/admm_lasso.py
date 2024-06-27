import cvxpy as cvx
import numpy as np
import pandas as pd
import abc as ABC

class LASSO(ABC): 
    def __init__(self inputs):
        A, x, b, rho, gamma = inputs

        # Assert A is a 2D numpy array
        assert isinstance(A, np.ndarray), "A must be a numpy array!"
        assert A.ndim == 2, "A must be a 2D numpy array!"

        # Assert x is a 2D numpy array with shape (n, 1)
        assert isinstance(x, np.ndarray), "x must be a numpy array!"
        assert x.ndim == 2, "x must be a 2D numpy array!"
        assert x.shape[1] == 1, "x must be a column vector (n, 1)!"

        # Assert b is a 2D numpy array with shape (m, 1)
        assert isinstance(b, np.ndarray), "b must be a numpy array!"
        assert b.ndim == 2, "b must be a 2D numpy array!"
        assert b.shape[1] == 1, "b must be a column vector (m, 1)!"

        # Assert rho is a float
        assert isinstance(rho, float), "rho must be a float!"

        # Assert gamma is a float
        assert isinstance(gamma, float), "gamma must be a float!"

        self._inputs = inputs
        self._xstop = None
        self._xstar = None
        self._xfinal = None
        self._pstop = None
        self._xstar = None
        self._xfinal = None
    
    @abstractmethod
    def solve(self):
        # Perform some calculation using A x and b and alter x.
        ...

    def get_stopping_point(self):
        # Returns (p, x) where stopping criterion is met
        return self._pstop, self._xstop,

    def get_star_point(self): 
        # Returns (p, x) at the smallest observed p 
        return self._pstar, self._xstar

    def get_final_point(self):
        # Returns (p, x) at the final iteration of algorithm
        return self._pfinal, self._xfinal
    
    
class CVXPY_ADMM(LASSO):
    def __init__(self, inputs):
        super()__init__(inputs)
        from multiprocessing import Pool
        NUM_PROCS = 4 

    def prox(args):
        f, v = args
        # x = Variable((n,1))
        f += (rho/2) * sum_squares(x - v)
        Problem(Minimize(f)).solve()
        return x.value

    def solve(self):
        A, x, b, rho, gamma = self._inputs
        funcs = [ cvx.sum_squares(A @ x - b), gamma * cvx.norm(x, 1) ]
        ui = [ np.zeros((n, 1)) for func in funcs ]
        xbar = np.zeros((n, 1))
        pool = Pool(NUM_PROCS)

        # ADMM loop.
        for i in range(50):
            prox_args = [ xbar - u for u in ui ]
            xi = pool.map(prox, zip(funcs, prox_args))
            xbar = sum(xi)/len(xi)
            ui = [ u + xavg - xbar

