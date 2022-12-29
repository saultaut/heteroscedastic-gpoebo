import torch
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

import sys

sys.path.append('../')
sys.path.append('../..')


from models.random_search import RANDOM_SEARCH

if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=10
    
    ackley = Ackley(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    ackley.bounds[0, :].fill_(-5)
    ackley.bounds[1, :].fill_(10)
    

    
    random_search = RANDOM_SEARCH(f=ackley, n_init=20, max_evals=200)
    
    X_init_shared = random_search.get_initial_points(NO_DIMENSIONS, 20)
    Y_init_shared = torch.tensor(
            [random_search.eval_objective_function(x, ackley) for x in X_init_shared]).unsqueeze(-1)
            
            
    running_time, fx, best_X, X_data, Y_data = random_search.optimize(X_init_shared, Y_init_shared)
    print(best_X)
    print(ackley(best_X))
            