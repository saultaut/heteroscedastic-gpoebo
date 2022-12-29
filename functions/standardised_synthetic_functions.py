import torch
import math

from torch import Tensor
from typing import List, Optional, Tuple

from torch.nn import Module
from abc import ABC, abstractmethod


class BaseNoisySyntheticTestProblem(Module, ABC):
    r"""Base class for test functions."""

    dim: int
    _bounds: List[Tuple[float, float]]
    _check_grad_at_opt: bool = True

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False,  heteroscedastic: Optional[str] = None, noise_multiplier: float = 1.0) -> None:
        r"""Base constructor for test functions.

        Arguments:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__()
        self.noise_std = noise_std
        self.noise_multiplier = noise_multiplier
        self.negate = negate
        self.heteroscedastic = heteroscedastic
        self.max_sphere_noise = self.heteroscedastic_sphere_noise(torch.ones(self.dim) * self._bounds[0][-1])
        self.max_rosenbrock_noise = self.heteroscedastic_rosenbrock_noise(torch.zeros(self.dim))
        self.min_rosenbrock_noise = self.heteroscedastic_rosenbrock_noise(torch.ones(self.dim) * 0.6)

        self.register_buffer(
            "bounds", torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)
        )

    def normalization(self, data):
        #normalize data between 0.1 and 0.5
        return (0.5-0.1) * (data - data.min()) / (data.max() + 0.001 - data.min()) + 0.1

    def heteroscedastic_sphere_noise(self, X: Tensor) -> Tensor:
        heter_noise =  torch.sum((X ** 2), dim=-1)
        return heter_noise
    
    def standartized_heteroscedastic_sphere_noise(self, X):
        #standartized data between 0.1 and 0.5
        noise = self.heteroscedastic_sphere_noise(X)
        # return (0.5-0.1) * (noise / self.max_sphere_noise) + 1.0
        return (noise / self.max_sphere_noise) * self.noise_multiplier #* self.dim
    
    def standartized_heteroscedastic_rosenbrock_noise(self, X):
        #standartized data between 0.1 and 0.5
        noise = self.heteroscedastic_rosenbrock_noise(X)
        return (0.5-0.1) * (noise - self.min_rosenbrock_noise) / (self.max_rosenbrock_noise - self.min_rosenbrock_noise) + 0.1
    
    def heteroscedastic_rosenbrock_noise(self, X: Tensor) -> Tensor:
        x_bar = 15.0 * X - 8.0
        inner_sum = torch.sum(100.0 * (x_bar[..., 1:] - x_bar[..., :-1] ** 2) ** 2 + (x_bar[..., :-1] - 1) ** 2, dim=-1)
        H = (inner_sum - 382700.0) / 375500.0
        
        return H       

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.

        Returns:
            A `batch_shape`-dim tensor ouf function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)

        if noise and self.heteroscedastic is not None:
            if self.heteroscedastic == 'Sphere':
                g = self.standartized_heteroscedastic_sphere_noise(X)
                f += g * torch.randn_like(f)
            elif self.heteroscedastic == 'Linear':
                if batch:
                    g = torch.mean(X, dim=-1) 
                else:
                    g = torch.mean(X, dim=0) 
                f -= g * 0.5 * torch.randn_like(f)
            else:
                g = self.standartized_heteroscedastic_rosenbrock_noise(X)
                f += g * torch.randn_like(f)

        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)


    @abstractmethod
    def evaluate_true(self, X: Tensor) -> Tensor:
        r"""Evaluate the function (w/o observation noise) on a set of points."""
        pass  # pragma: no cover

class SinWave(BaseNoisySyntheticTestProblem):
    """
    Standardised Sin wave function
    Optimal x value at: 8.0548
    Optimal y value at f(x) :
    """
    dim = 1
    _bounds = [(0.0, 10.0)]
    _optimal_value = None
    _optimizers = None

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sin(X) + 0.2 * X + 3
    

class NoisyBranin(BaseNoisySyntheticTestProblem):
    """
    Standardised 2D Branin Function from Picheny et al
    """
    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = None
    _optimizers = None

    def evaluate_true(self, X: Tensor) -> Tensor:
        x_bar_one = 15 * X[..., 0] - 5
        x_bar_two = 15 * X[..., 1]
        f = (1/51.95)*((x_bar_two - ((5.1*x_bar_one**2)/(4*math.pi**2)) + (5*x_bar_one/math.pi) - 6)**2 +
                           ((10 - 10/8*math.pi)*torch.cos(x_bar_one)) - 44.81)
        return f



class NoisyGoldstein_Price(BaseNoisySyntheticTestProblem):
    """
    Standardised Goldstein Price Function
    """
    dim = 2
    _bounds = [(0.0, 1.0), (0.0, 1.0)]
    _optimal_value = None
    _optimizers = None

    def evaluate_true(self, X: Tensor) -> Tensor:
        x_bar_one = 4 * X[..., 0] - 2
        x_bar_two = 4 * X[..., 1] - 2
        
       
        f = (1/2.427) * (torch.log((1 + (x_bar_one + x_bar_two + 1)**2*
                             (19 - 14*x_bar_one + 3*x_bar_one**2 - 14*x_bar_two + 6*x_bar_one*x_bar_two + 3*x_bar_two**2))*
                            (30 + (2*x_bar_one - 3*x_bar_two)**2*
                             (18 - 32*x_bar_one + 12*x_bar_one**2 + 48*x_bar_two - 36*x_bar_one*x_bar_two + 27*x_bar_two**2))) - 8.693)
        
        return f 

class NoisyHartmann(BaseNoisySyntheticTestProblem):
    r"""Hartmann synthetic test function.

    Most commonly used is the six-dimensional version (typically evaluated on
    `[0, 1]^6`):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at

        z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    with `H(z) = -3.32237`.
    """

    def __init__(
        self, dim=6, noise_std: Optional[float] = None, negate: bool = False,  heteroscedastic: Optional[str] = None, noise_multiplier: float = 1.0
    ) -> None:
        if dim not in (4, 6):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        # optimizers and optimal values for dim=4 not implemented
        optvals = None
        optimizers = None
        
        self._optimal_value = None
        self._optimizers = None
        super().__init__(noise_std=noise_std, negate=negate, heteroscedastic=heteroscedastic, noise_multiplier=noise_multiplier)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(1) - 0.0001 * self.P) ** 2, dim=2)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        elif self.dim == 6:
            H = -(2.58 - H) / 1.94
        return H

class NoisyRosenbrock(BaseNoisySyntheticTestProblem):
    r"""Rosenbrock synthetic test function.
    """

    def __init__(
        self, noise_std: Optional[float] = None, negate: bool = False,  heteroscedastic: Optional[str] = None, noise_multiplier: float = 1.0
    ) -> None:
        self.dim = 4
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = None
        super().__init__(noise_std=noise_std, negate=negate, heteroscedastic=heteroscedastic,noise_multiplier=noise_multiplier)

    def evaluate_true(self, X: Tensor) -> Tensor:
        x_bar = 15.0 * X - 5.0
        inner_sum = torch.sum(100.0 * (x_bar[..., 1:] - x_bar[..., :-1] ** 2) ** 2 + (x_bar[..., :-1] - 1) ** 2, dim=-1)
        H = (inner_sum - 382700.0) / 375500.0
        return H
    
class NoisySphere(BaseNoisySyntheticTestProblem):
    r"""
        The 6D Sphere function
    """

    def __init__(
        self, noise_std: Optional[float] = None, negate: bool = False,  heteroscedastic: Optional[str] = None, noise_multiplier: float = 1.0
    ) -> None:
        self.dim = 6
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = None
        super().__init__(noise_std=noise_std, negate=negate, heteroscedastic=heteroscedastic, noise_multiplier=noise_multiplier)

    def evaluate_true(self, X: Tensor) -> Tensor:
        inner_sum = torch.sum((X ** 2) * 2**torch.arange(1, 7), dim=-1)
        H = (inner_sum - 1745.0) / 899.0
        return H



class NoisyAckley10D(BaseNoisySyntheticTestProblem):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    def __init__(
        self, noise_std: Optional[float] = None, negate: bool = False, heteroscedastic: Optional[str] = None, noise_multiplier: float = 1.0
    ) -> None:
        self.dim = 10
        self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, heteroscedastic=heteroscedastic, noise_multiplier=noise_multiplier)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X: Tensor) -> Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        z = part1 + part2 + a + math.e
        return z
 
