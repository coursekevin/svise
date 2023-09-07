import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)


def difference(x1: Tensor, x2: Tensor) -> Tensor:
    r""" Computes the pairwise difference between two pairs of tensors

        :param torch.tensor(n,d) x1: first set of tensors
        :param torch.tensor(m,d) x2: second set of tensors

        :return: squared difference tensor (i,j) corresponds to the  ||x_i - x_j||^2     
        :rtype: torch.tensor(n,m,d)
    """
    return x1.unsqueeze(1) - x2.unsqueeze(0)


class SquaredExpKernel(nn.Module):
    r""" Squared exponential kernel extends torch.nn.Module base 

        :var torch.nn.parameter logsigf: log noise hyperparameter
        :var torch.nn.parameter loglen: log length scale hyperparameter
    """

    def __init__(self,):
        super(SquaredExpKernel, self).__init__()
        self.logsigf = Parameter(torch.Tensor(1))
        self.loglen = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r""" Resets the model parameters
        """
        # nn.init.normal_(self.loglen, mean=-4.0, std=1e-5)
        nn.init.normal_(self.loglen, mean=-2, std=0.1)  # -4.0 mean start
        nn.init.normal_(self.logsigf, mean=1e-2, std=1e-2)

    def get_natural_parameters(self) -> tuple:
        r""" Returns the parameters in their natural format

            :returns: (exp(loglen), exp(logsigf))
            :type: (torch.tensor, torch.tensor)
        """
        return (torch.exp(self.loglen), torch.exp(self.logsigf))

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        r""" Computes squared exponential kernel at x1 & x2

            :param torch.tensor(n,d) x1: first set of tensors
            :param torch.tensor(m,d) x2: second set of tensors

            :return: squared exp kernel
            :rtype: torch.tensor(n,m)
        """
        l, sigf = self.get_natural_parameters()
        diff = difference(x1, x2)
        sqrd = torch.pow(diff.sum(-1), 2)
        # return 1 / (l**2) * sigf**2 * torch.exp(-1 / (2 * l**2) * sqrd)
        return sigf ** 2 * torch.exp(-1 / (2 * l ** 2) * sqrd)

    def dkdt(self, t, tau):
        r""" Returns mixed derivative of kernel 

            :param torch.tensor(n,1) t: first one-dimensional input
            :param torch.tensor(m,1) tau: second one-dimensional input

            :return: time derivative of squared exp kernel 
            :rtype: torch.tensor(n,m)    
        
            .. warning::

                This function is only implmented when t and tau are 1D
        """
        l, sigf = self.get_natural_parameters()
        assert t.shape[-1] == 1
        assert tau.shape[-1] == 1
        diff = difference(t, tau).squeeze(-1)
        sqrd = torch.pow(diff, 2)
        # note added negative one, might need adjusting
        return -1 / (l ** 2) * sigf ** 2 * torch.exp(-1 / (2 * l ** 2) * sqrd) * diff

