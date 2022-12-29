import torch
from torch.distributions import Normal

from botorch.acquisition import ExpectedImprovement



def augmented_expected_improvement(model, mu_s, var_s, X_test, aleatoric_weight):
    T = mu_s
    best_effective_solution = T.max()
        
    EI = ExpectedImprovement(model=model, best_f=best_effective_solution)
        
    ei_values = EI(X_test.unsqueeze(1))
    
    #penalty
    noise = model.likelihood.noise.detach().sqrt().item()
    penalty = 1 - ( (aleatoric_weight * noise ) / torch.sqrt(var_s + aleatoric_weight * noise**2))
    ei_values = ei_values * penalty.squeeze(-1)
    
    
    best_index = ei_values.argmax()
    
    # get best prediction
    X_next = X_test[best_index, :].clone().unsqueeze(0)
    return X_next


def heteroscedastic_augmented_expected_improvement(model, mu, var, X_test, weight_matrix, aleatoric_weight):
    #---- EI ----
    T = mu #+ var.sqrt()
    best_effective_solution = T.max()
    
    mean = mu
    sigma = var.clamp_min(1e-9).sqrt()
    u = (mean - best_effective_solution.expand_as(mean)) / sigma

    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei_values_temp = sigma * (updf + u * ucdf)
    
    ### penalty
    l = model.likelihood(torch.zeros(X_test.shape[0]))
    noise_variance = l.variance.detach()
    noise_level = torch.sum(weight_matrix * noise_variance, axis=0)
    aleatoric_noise_std = noise_level.sqrt()
    
    penalty = 1 - ( (aleatoric_weight * aleatoric_noise_std ) / torch.sqrt(var + aleatoric_weight**2 * noise_level))
    ei_values = ei_values_temp * penalty.squeeze(-1)

    
    
    best_index = ei_values.argmax()
    # get best prediction
    X_next = X_test[best_index, :].clone().unsqueeze(0)
    # ###################
    return X_next

def heteroscedastic_noise_penalized_ei(model, mu, var, X_test, weight_matrix, aleatoric_weight):
    #---- EI ----
    T = mu #+ var.sqrt()
    best_effective_solution = T.max()
    
    mean = mu
    sigma = var.clamp_min(1e-9).sqrt()
    u = (mean - best_effective_solution.expand_as(mean)) / sigma

    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei_values_temp = sigma * (updf + u * ucdf)
    
    ### penalty
    l = model.likelihood(torch.zeros(X_test.shape[0]))
    noise_variance = l.variance.detach()
    noise_level = torch.sum(weight_matrix * noise_variance, axis=0)
    aleatoric_std = noise_level.sqrt()
    ei_values = ei_values_temp - aleatoric_weight*aleatoric_std
    #ei_values = ei_values_temp * self.aleatoric_weight - (1 - self.aleatoric_weight) * aleatoric_std
    
    
    best_index = ei_values.argmax()
    # get best prediction
    X_next = X_test[best_index, :].clone().unsqueeze(0)
    # ###################
    return X_next