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
        assert isinstance(x, np.ndarray), "x must be a numpy array!"
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
   
    
    def _objective_fn(self, A, x, b, rho, gamma):
        # f(x) = ||Ax - B||_2^2 + gamma * ||x||_1
        return self._loss_fn(A, x, b) + gamma * self._regularizer(x)

    def _loss_fn(self, A, x, b):
        return 0.5 * cx.sum_squares(A @ x + b) 
        # return cp.sum_of_squares(X @ beta - Y)

    def _regularizer(self, x):
        return cx.norm(x, 1) # L1 norm

    def _mse(self, A, x, b):
        return (1.0 / A.shape[0]) * self._loss_fn(A, x, b).value


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
    

# ABSTRACT CLASS 
class CVXPY_SOLVER(LASSO):
    def __init__(self, inputs, solver = cx.CLARABEL, **solver_kwargs):
        super().__init__(inputs)
        self._solver = solver
        self._solver_kwargs = solver_kwargs
        A, x, b, rho, gamma = inputs
        # Recreate x as Cvxpy Variable 
        x = inputs[1]
        x = cx.Variable((x.shape[0], 1))
        self._inputs = A, x, b, rho, gamma

    @abstractmethod
    def solve(self):
        pass
    

class CLARABEL(CVXPY_SOLVER):
    def solve(self): 
        A, x, b, rho, gamma = self._inputs
        # fis = [ cx.sum_squares(A @ x - b), gamma * cx.norm(x, 1) ]
        fis = [ self._loss_fn(A, x, b), gamma * self._regularizer(x) ]
        objective = cx.Minimize(sum(fis))
        problem = cx.Problem(objective)
        self._pstar = problem.solve(solver=self._solver, **self._solver_kwargs)
        self._xstar = x.value


class ADMM_PROX_POOL(CVXPY_SOLVER):
    # https://github.com/cvxpy/cvxpy/blob/master/examples/admm_lasso.py
    def __init__(self, inputs, **solver_kwargs):
        super().__init__(inputs)
        from multiprocessing import Pool
        self._NUM_PROCS = solver_kwargs.get('num_procs', 4) 
        self._max_iter = solver_kwargs.get('max_iter', 200)
        self._epsilon_abs = solver_kwargs.get('epsilon_abs', 1e-4)
        self._epsilon_rel = solver_kwargs.get('epsilon_rel', 1e-3)


    def _prox(self, args):
        x = self._inputs[1]
        rho = self._inputs[3]
        f, v, x = args
        problem = cx.Problem(cx.Minimize(f + (rho / 2) * cx.sum_squares(x - v)))
        problem.solve(solver=self._solver, **self._solver_kwargs)
        return x.value

    def solve(self):
        A, x, b, rho, gamma = self._inputs
        funcs = [ self._loss_fn(A, x, b), gamma * self._regularizer(x) ]
        ui = [ np.zeros((A.shape[1], 1)) for _ in funcs ]
        xbar = np.zeros((A.shape[1], 1))
        pool = Pool(self._NUM_PROCS)

        self._list_loss = []
        z_prev = xbar.copy()

        # BEGIN ADMM LOOP.
        for i in range(self._max_iter):
            prox_args = [ xbar - u for u in ui ] # Proximal arguments
            # Parallel proximal update
            xi = pool.map(self._prox, [ (func, prox_arg, x) for func, prox_arg in zip(funcs, prox_args) ])
            xbar = sum(xi) / len(xi) # Primal update (x avg across processes)
            ui = [ u + x_ - xbar for x_, u in zip(xi, ui) ] # Dual variable updates

            self._list_loss.append( (self._objective_fn(A, xbar, b, rho, gamma)).value )

            # BEGIN CHECK STOPPING CRITERION 
            if len(self._list_loss) > 1:
                # ADMM Boyd 3.12 stopping criterion
                r_k = np.linalg.norm(xbar - np.mean(xi, axis=0)) # primal residual
                s_k = np.linalg.norm(rho * (xbar - z_prev)) # dual residual
                epsilon_primal = np.sqrt(A.shape[1]) * self._epsilon_abs + \
                    self._epsilon_rel * max(np.linalg.norm(xbar), np.linalg.norm(np.mean(xi, axis=0)))#end-max 
                epsilon_dual = np.sqrt(A.shape[1]) * self._epsilon_abs + \
                    self._epsilon_rel * np.linalg.norm(ui[0]) 
                if r_k <= epsilon_primal and s_k <= epsilon_dual:
                    self._xstop = x.value
                    self._pstop = self._list_loss[-1]
                    # break
            # END CHECK STOPPING CRITERION
            z_prev = xbar.copy()
        # END ADMM LOOP

        self._xstar = xbar
        self._pstar = self._list_loss[-1]
    # end solve()


class ADMM_LAGRANGE_MPI(LASSO):
    def __init__(self, inputs, **solver_kwargs):
        super().__init__(inputs)
        import mpi4py.MPI as mpi
        self._comm = MPI.COMM_WORLD
        self._rank = comm.Get_rank() # Rank of the process
        self._size = comm.Get_size() # Number of processes
        self._max_iter = solver_kwargs.get('max_iter')

    def admm(self, A, b, x, z, y, rho, gamma):
        list_primal_res = []
        list_dual_res = []
        for k in range(0, self._max_iter): 
            ...
            pass
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

        comm.Broadcast(x, root=0)
        comm.Broadcast(rho, root=0)
        comm.Broadcast(gamma, root=0)

        admm(local_A, local_b, local_x, local_z, local_y, rho, gamma)
        


        ...









