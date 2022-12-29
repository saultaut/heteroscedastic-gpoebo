import torch
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

import sys

sys.path.append('../')
sys.path.append('../..')


from models.gpoebo_anpei import GPOEBO_ANPEI

if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=5
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
    
    gpoebo = GPOEBO_ANPEI(f=ackley, points_per_expert=20, n_init=40, aleatoric_weight = 0.1, max_evals=60, n_candidates=100, partition_type='random')
    running_time, fx, best_X, X_turbo, Y_turbo  = gpoebo.optimize()
    print(best_X)
    print(ackley(best_X))
            