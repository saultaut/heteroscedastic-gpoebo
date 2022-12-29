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

from botorch.models import FixedNoiseGP
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.acquisition import ExpectedImprovement


class BO_EI:
    def __init__(self,
                 f,
                 n_init=50,
                 max_evals=100,
                 n_candidates=5000
                ):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.n_init = n_init
        self.n_candidates = n_candidates
        
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

    def posterior(self, X_test):
        
        posterior = self.posterior_model.posterior(X_test)
        y_pred = self.posterior_model.likelihood(posterior.mvn)
        
        mu_s = y_pred.mean.cpu().detach()
        var_s = y_pred.variance.cpu().detach()
        
        mu = torch.reshape(mu_s, (-1, 1))
        var = torch.reshape(var_s, (-1, 1))

        return mu, var
    
    def find_best_X_value(self):
        """
        Find the best X value based on the best posterior mean value.

        Returns
        -------
        None.

        """
        
        X_test = self.get_initial_points(self.dim, self.n_candidates)
        mu, var =  self.posterior(X_test)
        best_mu_index = mu.argmax()
        best_X = X_test[best_mu_index, :].clone()
        
        return best_X
        

    def get_next_point(self, X_turbo_init=None, Y_turbo_init=None):
        
        """
        Args:
            X_turbo_init: shape Nxd
            Y_turbo_init: shape Nx1
        Return: 
            X_next: shape 1xd
        """
        
        X_turbo = X_turbo_init.clone()
        Y_turbo = Y_turbo_init.clone()        
    
        #normalize
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        # Fit a GP model
        try:
            model = self.get_fitted_model(X_turbo, train_Y)
        except Exception as ex:
            print(f"ERROR: {ex}")
            print("using untrained model")

        # EI
        X_test = self.get_initial_points(self.dim, self.n_candidates)
        posterior = model.posterior(X_test)
        
        mu_s = posterior.mean.cpu().detach()
        var_s = posterior.variance.cpu().detach()
        
        T = mu_s
        best_effective_solution = T.max()
        
        
        EI = ExpectedImprovement(model=model, best_f=best_effective_solution)
        
        ei_values = EI(X_test.unsqueeze(1))

        best_index = ei_values.argmax()
        # get best prediction
        X_next = X_test[best_index, :].clone().unsqueeze(0)       
        
        return X_next
                
                
    def optimize(self, X_turbo_init=None, Y_turbo_init=None):
		
        if X_turbo_init == None and Y_turbo_init == None:
            X_turbo = self.get_initial_points(self.dim, self.n_init)
            
            Y_turbo = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_turbo], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
        else:
        
            X_turbo = X_turbo_init.clone()
            Y_turbo = Y_turbo_init.clone()

        starttime = time.time() 
        for iteration in range(self.no_iterations):

            #normalize
            # train_Y = Y_turbo
            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
           # Fit a GP model
            try:
                model = self.get_fitted_model(X_turbo, train_Y)
            except Exception as ex:
                print(f"ERROR: {ex}")
                print("using untrained model")

                model = self.get_untrained_model(X_turbo, train_Y)
            
            
            # EI
            X_test = self.get_initial_points(self.dim, self.n_candidates)
            posterior = model.posterior(X_test)
            # y_pred = model.likelihood(posterior.mvn)
            
            mu_s = posterior.mean.cpu().detach()
            var_s = posterior.variance.cpu().detach()
            
            T = mu_s# + var_s.sqrt()
            best_effective_solution = T.max()
            
            
            EI = ExpectedImprovement(model=model, best_f=best_effective_solution)
            
            ei_values = EI(X_test.unsqueeze(1))

            best_index = ei_values.argmax()
            # get best prediction
            X_next = X_test[best_index, :].clone().unsqueeze(0)
            
            Y_next = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_next], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            
            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
            
            # Print current status
            print(
                f"{len(X_turbo)}) Best value: {Y_turbo.max()}"
            )
        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        #store the modeler for posterior prediction
        self.posterior_model = model
        
        running_time = endtime-starttime

        best_mu_index = mu_s.argmax()
        best_X = X_test[best_mu_index, :].clone()
        best_X = unnormalize(best_X, self.current_obj_fun.bounds)

        print(
            f"Best mu value: {mu_s.max()}, best_X: {best_X}"
        )
        fx = np.maximum.accumulate(Y_turbo.cpu())
        
        return running_time, fx, best_X,  X_turbo, Y_turbo 
       
if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=3
    rosenbrock = Rosenbrock(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    rosenbrock.bounds[0, :].fill_(-10)
    rosenbrock.bounds[1, :].fill_(10)
    
    levy = Levy(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    levy.bounds[0, :].fill_(-10)
    levy.bounds[1, :].fill_(10)
    
    rastrigin = Rastrigin(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    rastrigin.bounds[0, :].fill_(-5.12)
    rastrigin.bounds[1, :].fill_(5.12)
    
    ackley = Ackley(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    ackley.bounds[0, :].fill_(-5)
    ackley.bounds[1, :].fill_(10)
    
    obj_functions = {#"levy" : levy,
                     #"rastrigin" : rastrigin,
                      #"rosenbrock" : rosenbrock#,
                      "ackley": ackley
                     }
    
    bo = BO_EI(f=ackley, n_init=10, max_evals=30)
    # running_time, fx, best_X,  X_turbo, Y_turbo  = bo.optimize()
    # print(best_X)
    # print(ackley(best_X))
    
    X_init_shared = bo.get_initial_points(NO_DIMENSIONS, 20)
    Y_init_shared = torch.tensor(
            [bo.eval_objective_function(x, ackley) for x in X_init_shared]).unsqueeze(-1)
    
    x_next = bo.get_next_point(X_init_shared, Y_init_shared)
    print(x_next)
    print(bo.eval_objective_function(x_next, ackley))

            

            

        
        
        