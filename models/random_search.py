import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
import gpytorch
import numpy as np
import time
import math

class RANDOM_SEARCH:
    def __init__(self,
                 f,
                 n_init=50,
                 max_evals=100):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.n_init = n_init
        self.no_iterations = max_evals # we run from the start

        
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def eval_objective_function(self, x, obj_fun):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return obj_fun(unnormalize(x, obj_fun.bounds))

    def get_initial_points(self, dim, n_pts):
        sobol = SobolEngine(dimension=dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init
        
    def optimize(self, X_data_init=None, Y_data_init=None): #X_data_init
        """
        X_data_init: shape Nxd
        Y_data_init: shape Nx1
        """
		
        if X_data_init == None and Y_data_init == None:
            X_data = self.get_initial_points(self.dim, self.n_init)
            
            Y_data = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_data], dtype=self.dtype, device=self.device
            )
        else:
            assert X_data_init.dim() == 2 and Y_data_init.dim() == 2
            X_data = X_data_init.clone()
            Y_data = Y_data_init.clone()
            
        starttime = time.time()

        for iteration in range(self.no_iterations):

            X_next = self.get_initial_points(self.dim, 1)
            Y_next = self.eval_objective_function(X_next, self.current_obj_fun).unsqueeze(-1)

            
            X_data = torch.cat((X_data, X_next), dim=0)
            Y_data = torch.cat((Y_data, Y_next), dim=0)


        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime

        best_y_index = Y_data.argmax()
        best_X = X_data[best_y_index, :].clone()
        best_X = unnormalize(best_X, self.current_obj_fun.bounds)

        fx = np.maximum.accumulate(Y_data.cpu())
        
        return running_time, fx, best_X,  X_data, Y_data 
       

            

            

            

        
        
        