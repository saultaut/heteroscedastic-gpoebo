import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
import numpy as np
import time

from acquisition.acquisition_functions import augmented_expected_improvement

import sys
sys.path.append('../')

class BO_AEI:
    def __init__(self,
                 f,
                 n_init=50,
                 max_evals=100,
                 aleatoric_weight = 1.0,
                 n_candidates=5000
                ):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.aleatoric_weight = torch.tensor(aleatoric_weight)
        
        self.no_iterations = max_evals
        
        self.device = torch.device("cpu")
        self.dtype = torch.float
        
    def get_initial_points(self, dim, n_pts):
        sobol = SobolEngine(dimension=dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init

    def eval_objective_function(self, x, obj_fun):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return obj_fun(unnormalize(x, obj_fun.bounds))

    def get_fitted_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-3))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
        return model

    def get_untrained_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        return model

    def optimize(self, X_data_init=None, Y_data_init=None):
		
        if X_data_init == None and Y_data_init == None:
            X_data = self.get_initial_points(self.dim, self.n_init)
            
            Y_data = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_data], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
        else:
        
            X_data = X_data_init.clone()
            Y_data = Y_data_init.clone()



        starttime = time.time() 
        for iteration in range(self.no_iterations):

            #normalize
            train_Y = (Y_data - Y_data.mean()) / Y_data.std()
           # Fit a GP model
            try:
                model = self.get_fitted_model(X_data, train_Y)
            except Exception as ex:
                print(f"ERROR: {ex}")
                print("using untrained model")

                model = self.get_untrained_model(X_data, train_Y)
            
            
            # AEI
            X_test = self.get_initial_points(self.dim, self.n_candidates)
            posterior = model.posterior(X_test)
            
            mu_s = posterior.mean.cpu().detach()
            var_s = posterior.variance.cpu().detach()
        
            
            X_next = augmented_expected_improvement(model, mu_s, var_s, X_test, self.aleatoric_weight)
            Y_next = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_next], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            
            # Append data
            X_data = torch.cat((X_data, X_next), dim=0)
            Y_data = torch.cat((Y_data, Y_next), dim=0)
            
            # Print current status
            print(
                f"{len(X_data)}) Best value: {Y_data.max()}"
            )
        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime

        best_mu_index = mu_s.argmax()
        best_X = X_test[best_mu_index, :].clone()
        best_X = unnormalize(best_X, self.current_obj_fun.bounds)

        print(
            f"Best mu value: {mu_s.max()}, best_X: {best_X}"
        )
        fx = np.maximum.accumulate(Y_data.cpu())
        
        return running_time, fx, best_X,  X_data, Y_data 

            

            

        
        
        