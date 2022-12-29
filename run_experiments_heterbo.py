import torch
from models.bo_aei import BO_AEI
from models.bo_ei import BO_EI
from models.gpoebo_haei import GPOEBO_HAEI
from models.gpoebo_anpei import GPOEBO_ANPEI
from models.random_search import RANDOM_SEARCH
import traceback
import os

import numpy as np
import random

from functions.standardised_synthetic_functions import NoisyBranin, NoisyGoldstein_Price, NoisyHartmann, NoisyRosenbrock, NoisySphere
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import pickle as pkl


def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts)
    return X_init

def eval_objective_function(x, obj_fun, noise=True):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return obj_fun(unnormalize(x, obj_fun.bounds), noise=noise)
    
def get_initial_dataset(n_init, f, no_trials, function_name='', load_from_file=False):
    if load_from_file:
        currentDirectory = os.getcwd()
        filename = currentDirectory + "/results/init_dataset_" + function_name +".pkl"
        with open(filename, 'rb') as handle:
            init_data = pkl.load(handle)
    else:
        dim = f.dim
        init_data = []
        for i in range(no_trials):
            
            X_init_shared = get_initial_points(dim, n_init)
            Y_init_shared = torch.tensor(
                [eval_objective_function(x, f) for x in X_init_shared], dtype=dtype, device=device
            ).unsqueeze(-1)
            
            init_data.append((X_init_shared, Y_init_shared))
            
    return init_data

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    

    noise_multiplier = 1.0
    heteroscedastic_noise = 'Sphere'
    noise_std = None  


    noisyBranin = NoisyBranin(negate=True, noise_std = noise_std, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)  
    noisyGoldstein_Price = NoisyGoldstein_Price(negate=True, noise_std = noise_std, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)
    noisyHartmann4D = NoisyHartmann(dim=4, noise_std = noise_std, negate=True, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)
    noisyHartmann6D = NoisyHartmann(dim=6, noise_std = noise_std, negate=True, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)
    noisyRosenbrock = NoisyRosenbrock(negate=True, noise_std = noise_std, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)
    noisySphere = NoisySphere(negate=True, noise_std = noise_std, heteroscedastic=heteroscedastic_noise, noise_multiplier=noise_multiplier)


    obj_functions = {
                        "NoisyBranin_" + str(heteroscedastic_noise): noisyBranin,
                        # "NoisyGoldsteinPrice_" + str(heteroscedastic_noise): noisyGoldstein_Price,
                      # "NoisyHartmann4D_" + str(heteroscedastic_noise): noisyHartmann4D,
                      # "NoisyHartmann6D_" + str(heteroscedastic_noise): noisyHartmann6D,
                      # "NoisyRosenbrock_" + str(heteroscedastic_noise): noisyRosenbrock,
                      # "NoisySphere_" + str(heteroscedastic_noise): noisySphere,
                     }

    model_list = ["gpoebo_haei"]

    save_to_file = False
    load_from_file = False
    
    for function_name, opt_function in obj_functions.items():
          print(function_name)
    
          max_evals = 5 * opt_function.dim
          n_init = 10 * opt_function.dim
          points_per_expert = 4 * opt_function.dim
          n_candidates = 500
          aleatoric_weight = 0.1 if opt_function.dim < 3 else 0.5
          partition_type = 'random' 
    
          N_TRIALS = 50
          
          init_dataset = get_initial_dataset(n_init, opt_function, N_TRIALS, function_name=function_name, load_from_file=load_from_file)
          
    
          for model_name in model_list:
              
              time_history = []
              opt_history = []
              best_noisy_free_Y_history = []
              
              for trial in range(N_TRIALS):
                  set_random_seed(trial)
                  
                  
                  models = {
                    "gpoebo_haei" : GPOEBO_HAEI(f=opt_function, points_per_expert=points_per_expert, aleatoric_weight = aleatoric_weight, n_init=n_init, max_evals=max_evals, n_candidates=n_candidates, partition_type=partition_type),
                    "gpoebo_anpei" : GPOEBO_ANPEI(f=opt_function, points_per_expert=points_per_expert, aleatoric_weight = aleatoric_weight, n_init=n_init, max_evals=max_evals, n_candidates=n_candidates, partition_type=partition_type),
                    "bo_AEI" : BO_AEI(f=opt_function, n_init=n_init, aleatoric_weight = aleatoric_weight, max_evals=max_evals,n_candidates=n_candidates),
                    "bo_EI" : BO_EI(f=opt_function, n_init=n_init, max_evals=max_evals,n_candidates=n_candidates),
                    "Random_Search" : RANDOM_SEARCH(f=opt_function, n_init=n_init, max_evals=max_evals)
                    }
    
                  print(f"Model: {model_name} Trial : {str(trial)}")
                  model = models[model_name]
    
                  try:
                    X_init_shared, Y_init_shared = init_dataset[trial]
                    running_time, fx, best_X, X_turbo, Y_turbo  = model.optimize(X_init_shared, Y_init_shared)
                    
                    #find the best noise free point
                    Y_noise_free_list = torch.tensor([eval_objective_function(x, opt_function, noise=False) for x in X_turbo[n_init:]])
                    Y_noise_free = Y_noise_free_list.max()
            
                    fx = np.maximum.accumulate(Y_noise_free_list.cpu())
                    opt_history.append(fx.numpy())
                    best_noisy_free_Y_history.append(Y_noise_free.item())
                  except Exception as ex:
                    print(f"ERROR in model: {model_name} with ex {str(ex)}")
                    traceback.print_exc()            
              
                  
              results = np.array(opt_history, dtype=np.float64)
              means = np.mean(results, axis=0)
              stds = np.std(results, axis=0)
              
              print('Average values: ' + str(means))
              print('Standard deviation: ' + str(stds))
            

              
