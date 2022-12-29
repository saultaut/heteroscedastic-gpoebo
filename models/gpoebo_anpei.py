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

from torch.distributions import Normal


from sklearn.cluster import KMeans 
from sklearn.neighbors import BallTree  

from acquisition.acquisition_functions import heteroscedastic_noise_penalized_ei

import sys
sys.path.append('../')

class GPOEBO_ANPEI:
    def __init__(self,
                 f,
                 points_per_expert = 20,
                 weighting = 'diff_entr',
                 n_init=50,
                 max_evals=100,
                 aleatoric_weight = 1.0,
                 n_candidates=5000,
                 partition_type='random'):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.beta = torch.tensor(1.0)
        self.weighting = weighting
        self.n_init = n_init  # 2*dim, which corresponds to 5 batches of 4
        self.n_candidates = n_candidates
        self.aleatoric_weight = torch.tensor(aleatoric_weight)
        
        self.POINTS_PER_EXPERT = points_per_expert

        self.partition_type = partition_type

        
        self.no_iterations = max_evals
        
        self.device = torch.device("cpu")
        self.dtype = torch.float
        
        #posterior stuff
        self.model = None
        self.N_EXPERTS = None
        self.weight_matrix = None

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
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
        return model

    def get_unfitted_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
        return model

    def normalize_weights(self, weight_matrix):
        """ Compute unnormalized weight matrix
        """
        
        sum_weights = torch.sum(weight_matrix, axis=0)
        weight_matrix = weight_matrix / sum_weights
        
        return weight_matrix

    def compute_weights(self, mu_s, var_s, weighting, prior_var=None, softmax=False, power=10):
        
        """ Compute unnormalized weight matrix
        """
        
        if weighting == 'uniform':
            weight_matrix = torch.ones(mu_s.shape) / mu_s.shape[0]
    
        if weighting == 'diff_entr':
            weight_matrix = 0.5 * (torch.log(prior_var) - torch.log(var_s))
    
        if weighting == 'variance':
            weight_matrix = torch.exp(-power * var_s)
            
        if weighting == 'no_weights':
            weight_matrix = 1
        
        return weight_matrix
    
    def posterior(self, X_test):
        
        if X_test.shape[0] != self.n_candidates:
            raise Exception("sizes are not the same")
            
        #these values can be pre-allcoated at the begining with maximum size values, but efficiency is not clear.
        mu_s = torch.zeros(self.N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)
        var_s = torch.zeros(self.N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)
        prior = torch.zeros(self.N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)

        try:
            # get prior
            with gpytorch.settings.prior_mode(True):
                #y_pred = f + noise
                y_prior = self.model(X_test)        
                prior = y_prior.variance.detach()   #torch.Size([1, 5000])
            
            #get posterior
            posterior = self.model.posterior(X_test)
            # y_pred = self.model.likelihood(posterior.mvn)
            
            mu_s = posterior.mean.cpu().detach()
            var_s = posterior.variance.cpu().detach() #torch.Size([1, 5000, 1])
            
        except Exception as e:
            print("Oops!", e.__class__, "occurred.")
        
        var_s = var_s.squeeze(-1) 
        mu_s = mu_s.squeeze(-1)
        weight_matrix = self.compute_weights(mu_s, var_s, weighting=self.weighting, prior_var=prior, softmax=False)
        del prior
        # Compute individual precisions - dim: n_experts x n_test_points
        prec_s = 1/var_s
        self.weight_matrix = self.normalize_weights(weight_matrix)
    
        prec = torch.sum(self.weight_matrix * prec_s, axis=0)
        
                        
        var = 1 / prec
        mu = var * torch.sum(self.weight_matrix * prec_s * mu_s, axis=0)
        
        del mu_s, var_s, 
        
        # mu = torch.reshape(mu, (-1, 1))
        # var = torch.reshape(var, (-1, 1))

        return mu, var

    def optimize(self, X_data_init=None, Y_data_init=None):
        """

        Parameters
        ----------
        X_data_init : Tensor, shape Nxd 
        Y_data_init : Tensor, shape Nx1

        Returns
        -------
        running_time : TYPE
            DESCRIPTION.
        fx : TYPE
            DESCRIPTION.
        best_X : TYPE
            DESCRIPTION.
        X_data : TYPE
            DESCRIPTION.
        Y_data : TYPE
            DESCRIPTION.

        """
        
		
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
            # Compute number of experts 
            self.N_EXPERTS = int(np.max([X_data.shape[0] / self.POINTS_PER_EXPERT, 1]))
            # Compute number of points experts 
            N = int(X_data.shape[0] / self.N_EXPERTS)
            # If random partition, assign random subsets of data to each expert

            # train_Y = Y_data
            train_Y = (Y_data - Y_data.mean()) / Y_data.std()

            partition = []
            if self.partition_type == 'random':
                partition = np.random.choice(X_data.shape[0], size=(self.N_EXPERTS, N), replace=False)
                partition = np.array([np.random.choice(X_data.shape[0], N, replace=False) for i in range(self.N_EXPERTS)])
            elif self.partition_type == 'SoD':
                partition = np.random.choice(X_data.shape[0], size=(self.N_EXPERTS, N), replace=True)
            elif self.partition_type == 'kmeans':
                kmeans = KMeans(n_clusters=self.N_EXPERTS)  
                tmp = X_data.numpy()
                kmeans = kmeans.fit(tmp)
                #plabel = kmeans.predict(tmp)
                centers = kmeans.cluster_centers_
                tree = BallTree(X_data)
                dist, partition = tree.query(centers, k=N)
                
            
            if iteration % 20 == 0:
                print(f'Number of experts:{self.N_EXPERTS}')
                
            batched_train_X = torch.stack([X_data[partition[k]] for k in range(self.N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            batched_train_Y = torch.stack([train_Y[partition[k]] for k in range(self.N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            
            try:
                self.model = self.get_fitted_model(batched_train_X, batched_train_Y)
            except Exception as ex:
                print(f"ERROR: {ex}")
                print("using untrained model")


            self.model = self.model.to(device=self.device, dtype=self.dtype)
            
            ##### posterior prediction and sampling ####
			
            X_test = self.get_initial_points(self.dim, self.n_candidates)
            mu, var = self.posterior(X_test)
            
            assert mu.dim() == 1
            X_next = heteroscedastic_noise_penalized_ei(self.model, mu, var, X_test, self.weight_matrix, self.aleatoric_weight)
           

            Y_next = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_next], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
        
            # Append data
            X_data = torch.cat((X_data, X_next), dim=0) # shape Nxd
            Y_data = torch.cat((Y_data, Y_next), dim=0) # shape Nx1
            

            # Print current status
            print(
                f"{len(X_data)}) Best Y value: {Y_data.max()}"
            )
        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime
        
        ### best mean value
        best_mu_index = mu.argmax()
        best_X = X_test[best_mu_index, :].clone()
            
        best_X = unnormalize(best_X, self.current_obj_fun.bounds)

        print(
            f"Best mu value: {mu.max()}, best_X: {best_X}"
        )
            
        fx = np.maximum.accumulate(Y_data.cpu())
        
        return running_time, fx, best_X, X_data, Y_data   
       

            

            

            

        
        
        