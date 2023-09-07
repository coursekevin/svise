import numpy.testing as npt
import torch
from torch import Tensor
from svise.variationalsparsebayes import *
from torch.nn import Sigmoid
from torch.distributions import MultivariateNormal


def test_bayesian_linear():
    in_features = 3
    out_features = 4
    batch_size = 5
    x = torch.randn(batch_size, in_features)
    sigma = Sigmoid()
    bl = BayesianLinear(
        in_features, out_features, tau=1e-5, n_reparam=20, nonlinearity=sigma
    )
    rl = BayesianResidual(out_features, tau=1e-5, n_reparam=20, nonlinearity=sigma)
    bl.reparam_sample()
    y1 = bl(x)
    y1_known = sigma(x @ bl.w.transpose(-2, -1) + bl.b.unsqueeze(1))
    y2 = rl(y1)
    y2_known = sigma(y1_known @ rl.w.transpose(-2, -1) + rl.b.unsqueeze(1)) + y1_known
    # testing outputs are as expected
    with torch.no_grad():
        npt.assert_array_equal(y1_known, y1)
        npt.assert_array_equal(y2_known, y2)
    # checking kl divergence
    with torch.no_grad():
        npt.assert_array_equal(bl.kl_divergence, bl.prior.kl_divergence())
        npt.assert_array_equal(rl.kl_divergence, rl.prior.kl_divergence())


def test_sparse_bnn():
    model = SparseBNN(3, 4, n_layers=1)
    kl_true = 0.0
    for layer in model.layers:
        kl_true += layer.kl_divergence
    with torch.no_grad():
        npt.assert_array_equal(kl_true, model.kl_divergence)


def test_sparse_bnn_loglikelihood():
    model = SparseBNN(3, 4, n_layers=1)
    n_batch = 5
    x = torch.randn(n_batch, 3)
    y_true = torch.randn(n_batch, 4)
    y_pred = model(x)
    log_prob = 0
    for j in range(n_batch):
        mvn = MultivariateNormal(y_true[j], torch.diag(model.sigma.pow(2)))
        log_prob += mvn.log_prob(y_pred[:, j]).mean()
    log_prob /= n_batch
    with torch.no_grad():
        npt.assert_almost_equal(log_prob, model.log_likelihood(x, y_true))

