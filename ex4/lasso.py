import cvxpy as cx
import numpy as np
from multiprocessing import Pool
from abc import ABC, abstractmethod

class LASSO(ABC): 
    def __init__(self, inputs):
        A, x, b, rho, gamma = inputs 

        # Assert A is a 2D numpy array
        assert isinstance(A, np.ndarray), "A must be a numpy array!"
        assert A.ndim == 2, "A must be a 2D numpy array!"

        # Assert x is a 2D numpy array with shape (n, 1)
        assert isinstance(x, cx.Variable), "x must be a cvxpy Variable!"
        assert x.ndim == 2, "x must be a 2D Variable!"
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
        self._list_loss = []

        self._xstar = None
        self._xstop = None
        self._xfinal = None
        self._pstar = None
        self._pstop = None
        self._pfinal = None
   
    def objective_fn(A, x, b, rho, gamma):
        # f(x) = ||Ax - B||_2^2 + gamma * ||x||_1
        return loss_fn(A, x, b) + gamma * regularizer(x)

    def loss_fn(A, x, b):
        return 0.5 * cx.sum_squares(A @ x, b) 
        # return cp.sum_of_squares(X @ beta - Y)

    def regularizer(x):
        return cx.norm(x, 1) # L1 norm

    def mse(A, x, b):
        return (1.0 / A.shape[0]) * loss_fn(A, x, b).value

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
    

class CVXPY_CLARABEL(LASSO):
    def __init__(self, inputs, solver=cx.CLARABEL, **solver_kwargs):
        super().__init__(inputs)
        self._solver = solver
        self._solver_kwargs = solver_kwargs

    def solve(self): 
        A, x, b, rho, gamma = self._inputs
        fis = [ cx.sum_squares(A @ x - b), gamma * cx.norm(x, 1) ]
        objective = cx.Minimize(sum(fis))
        problem = cx.Problem(objective)
        self._pstar = problem.solve(solver=self._solver, **self._solver_kwargs)
        self._xstar = x.value


class CVXPY_ADMM_PROX_POOL(LASSO):
    """
    This is a weird implementation bc it's ADMM but also proximal operator.
    It doesn't even utilize dual decomposition to utilize all 4 processes,
    just maps the least squares term and regularizer term to 2 processes.
    https://github.com/cvxpy/cvxpy/blob/master/examples/admm_lasso.py
    """
    def __init__(self, inputs):
        super().__init__(inputs)
        from multiprocessing import Pool
        self._NUM_PROCS = 4 

    def _prox(self, args):
        x = self._inputs[1]
        rho = self._inputs[3]
        f, v, x = args
        problem = cx.Problem(cx.Minimize(f + (rho / 2) * cx.sum_squares(x - v)))
        problem.solve()
        return x.value

    def solve(self):
        A, x, b, rho, gamma = self._inputs
        funcs = [ cx.sum_squares(A @ x - b), gamma * cx.norm(x, 1) ]
        ui = [ np.zeros((A.shape[1], 1)) for _ in funcs ]
        xbar = np.zeros((A.shape[1], 1))
        pool = Pool(self._NUM_PROCS)

        self._list_loss = []

        # ADMM loop.
        for i in range(50):
            prox_args = [ xbar - u for u in ui ]
            xi = pool.map(self._prox, [ (func, prox_arg, x) for func, prox_arg in zip(funcs, prox_args) ])
            xbar = sum(xi) / len(xi)
            ui = [ u + x_ - xbar for x_, u in zip(xi, ui) ]
            self._list_loss.append( (cx.sum_squares(A @ xbar - b) + gamma * cx.norm(xbar, 1)).value )

        self._xstar = x.value
        self._pstar = self._list_loss[-1]


class ADMM_MPI(LASSO):
    def __init__(self, inputs):
        super().__init__(inputs)
        import mpi4py.MPI as mpi
        self._comm = MPI.COMM_WORLD
        self._rank = comm.Get_rank() # Rank of the process
        self._size = comm.Get_size() # Number of processes

    def admm(self, A, b, x, z, y):
        ... 
        pass

        
        

    def solve(self):
        if self._rank == 0:
            A, x, b, rho, gamma = self._inputs 
        else:
            A = None
            x = None
            b = None
            rho = None
            gamma = None

        Mi = A.shape[0] // self._size
        N = A.shape[1]

        local_A = np.empty((Mi, N), dtype='float64') 
        local_b = np.empty(Mi, dtype='float64')
        local_x = np.zeros(N, dtype='float64')
        local_z = np.zeros(N, dtype='float64')
        local_y = np.zeros(N, dtype='float64')

        comm.Scatter(A, local_A, root=0) 
        comm.Scatter(b, local_b, root=0)

        admm(local_A, local_b, local_x, local_z, local_y)
        


        ...









