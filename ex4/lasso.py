import cvxpy as cvx
import numpy as np
from multiprocessing import Pool
from abc import ABC, abstractmethod

class LASSO(ABC): 
    def __init__(self, inputs):
        self._inputs = inputs
        self._A, self._x, self._b, self._rho, self._gamma = inputs

        # Assert A is a 2D numpy array
        assert isinstance(self._A, np.ndarray), "A must be a numpy array!"
        assert self._A.ndim == 2, "A must be a 2D numpy array!"

        # Assert x is a 2D numpy array with shape (n, 1)
        assert isinstance(self._x, cvx.Variable), "x must be a cvxpy Variable!"
        assert self._x.ndim == 2, "x must be a 2D Variable!"
        assert self._x.shape[1] == 1, "x must be a column vector (n, 1)!"

        # Assert b is a 2D numpy array with shape (m, 1)
        assert isinstance(self._b, np.ndarray), "b must be a numpy array!"
        assert self._b.ndim == 2, "b must be a 2D numpy array!"
        assert self._b.shape[1] == 1, "b must be a column vector (m, 1)!"

        # Assert rho is a float
        assert isinstance(self._rho, float), "rho must be a float!"

        # Assert gamma is a float
        assert isinstance(self._gamma, float), "gamma must be a float!"

        self._xstar = None
        self._xstop = None
        self._xfinal = None
        self._pstar = None
        self._pstop = None
        self._pfinal = None
    
    @abstractmethod
    def solve(self):
        pass

    def get_star_point(self): 
        # Returns (p, x) at the smallest observed p 
        return self._pstar, self._xstar

    def get_stop_point(self):
        # Returns (p, x) where stopping criterion is met
        return self._pstop, self._xstop,

    def get_final_point(self):
        # Returns (p, x) at the final iteration of algorithm
        return self._pfinal, self._xfinal
    

class CVXPY_SOLVE(LASSO):
    def solve(self):
        A, x, b, rho, gamma = self._inputs
        fis = [ cvx.sum_squares(A @ x - b), gamma * cvx.norm(x, 1) ]
        objective = cvx.Minimize(sum(fis))
        problem = cvx.Problem(objective)
        self._pstar = problem.solve()
        self._xstar = x.value


class CVXPY_ADMM_POOL(LASSO):
    def __init__(self, inputs):
        super().__init__(inputs)
        from multiprocessing import Pool
        self._NUM_PROCS = 4 

    def _prox(self, args):
        x = self._inputs[1]
        rho = self._inputs[3]
        f, v, x = args
        problem = cvx.Problem(cvx.Minimize(f + (rho / 2) * cvx.sum_squares(x - v)))
        problem.solve()
        return x.value

    def solve(self):
        A, x, b, rho, gamma = self._inputs
        funcs = [ cvx.sum_squares(A @ x - b), gamma * cvx.norm(x, 1) ]
        ui = [np.zeros((A.shape[1], 1)) for _ in funcs]
        xbar = np.zeros((A.shape[1], 1))
        pool = Pool(self._NUM_PROCS)

        list_loss = []

        # ADMM loop.
        for i in range(50):
            prox_args = [ xbar - u for u in ui ]
            xi = pool.map(self._prox, [ (func, prox_arg, x) for func, prox_arg in zip(funcs, prox_args) ])
            xbar = sum(xi) / len(xi)
            ui = [ u + x_ - xbar for x_, u in zip(xi, ui) ]
            list_loss.append( (cvx.sum_squares(A @ xbar - b) + gamma * cvx.norm(xbar, 1)).value )

        self._xstar = x.value
        self._pstar = list_loss[-1]


class ADMM_MPI(LASSO):
    def __init__(self, inputs):
        super().__init__(inputs)
        import mpi4py.MPI as mpi
        self._comm = MPI.COMM_WORLD
        self._rank = comm.Get_rank() # Rank of the process
        self._size = comm.Get_size() # Number of processes

    def admm(self):
        pass

    def solve(self):
        pass
        









