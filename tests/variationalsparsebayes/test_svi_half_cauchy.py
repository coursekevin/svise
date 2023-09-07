import numpy.testing as npt
import torch
from torch import Tensor
from torch.distributions.half_cauchy import HalfCauchy
from torch.distributions.gamma import Gamma
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
import torch.nn as nn
from scipy.stats import invgamma
from svise.variationalsparsebayes import *

torch.set_default_dtype(torch.float64)


def test_normal_mean_field_variational():
    """
    This test checks that the mean field variational distribution is correctly generating
    reparameterized samples by checking the stastics of the samples against a MC estimate
    """
    mu_exact = 2 * torch.ones(10)
    sigma = torch.ones(10) * 3
    log_sigma = torch.log(sigma)
    # testing reparameterization
    distn = NormalMeanFieldVariational(mu_exact, log_sigma)
    samples = distn(200000)
    mu_est = samples.mean(0)
    std_est = samples.std(0)
    with torch.no_grad():
        npt.assert_array_almost_equal(mu_est, mu_exact, decimal=1)
        npt.assert_array_almost_equal(std_est, torch.exp(log_sigma), decimal=1)
        # testing variance calculation
        npt.assert_array_equal(distn.var(), torch.exp(log_sigma).pow(2))


def test_log_normal_mean_field_variational():
    """
    This test checks that the log normal mean field variational distribution 
    is correctly generating reparameterized samples by checking the stastics of the samples against a MC estimate
    """
    mu = 2 * torch.ones(10)
    sigma = torch.ones(10) * 1e-1
    log_sigma = torch.log(sigma)
    # testing reparameterization
    distn = LogNormalMeanFieldVariational(mu, log_sigma)
    mean_exact = torch.exp(mu + sigma.pow(2) / 2)
    var_exact = (torch.exp(sigma ** 2) - 1) * torch.exp(2 * mu + sigma ** 2)
    samples = distn(200000)
    mean_est = samples.mean(0)
    var_est = samples.var(0)
    with torch.no_grad():
        npt.assert_array_almost_equal(mean_est, mean_exact, decimal=1)
        npt.assert_array_almost_equal(var_est, var_exact, decimal=1)
        # testing variance calculation
        npt.assert_array_equal(distn.var(), torch.exp(log_sigma).pow(2))


def test_half_cauchy_prior_log_normal_reparm():
    prior = SVIHalfCauchyPrior(10, 1e-5)
    s_a = prior.s_a
    s_b = prior.s_b
    # checking naive
    naive_samples = torch.sqrt(s_a(20000) * s_b(20000))
    # checking reparam
    samples = prior._log_normal_reparam(
        20000, s_a.d, s_a.mu, s_b.mu, s_a.log_sigma, s_b.log_sigma
    )
    mu = 0.5 * (s_a.mu + s_b.mu)
    sigma = torch.sqrt(0.25 * (s_a.var() + s_b.var()))
    mean_exact = torch.exp(mu + sigma.pow(2) / 2)
    var_exact = (torch.exp(sigma ** 2) - 1) * torch.exp(2 * mu + sigma ** 2)
    # checking
    with torch.no_grad():
        npt.assert_array_almost_equal(mean_exact, samples.mean(0), decimal=2)
        npt.assert_array_almost_equal(mean_exact, naive_samples.mean(0), decimal=2)
        npt.assert_array_almost_equal(var_exact, samples.var(0), decimal=2)
        npt.assert_array_almost_equal(var_exact, naive_samples.var(0), decimal=2)


def test_half_cauchy_prior_init():
    """
    This test confirms that custom initialization is working correctly
    """
    d = 10
    prior = SVIHalfCauchyPrior(d, 1e-5, w_init=torch.ones(d))
    w = prior.get_reparam_weights(1).flatten()
    with torch.no_grad():
        npt.assert_array_almost_equal(torch.ones(d), w, decimal=2)


def test_half_cauchy_prior_get_reparam_weights():
    """
    This test confirms that the fast "local-reparameterization trick" is generating
    samples from the variational distribution which match with a naive hierarchical 
    sampling scheme
    """
    prior = SVIHalfCauchyPrior(3, 1e-5)
    n_samples = 20000
    w = prior.get_reparam_weights(n_samples)
    s_a = LogNormal(prior.s_a.mu, prior.s_a.var().sqrt())
    s_b = LogNormal(prior.s_b.mu, prior.s_b.var().sqrt())
    gamma_a = LogNormal(prior.gamma_a.mu, prior.gamma_a.var().sqrt())
    gamma_b = LogNormal(prior.gamma_b.mu, prior.gamma_b.var().sqrt())
    wi = Normal(prior.w_tilde.mu, prior.w_tilde.var().sqrt())
    w_true = wi.sample((n_samples,)) * torch.sqrt(
        s_a.sample((n_samples,))
        * s_b.sample((n_samples,))
        * gamma_a.sample((n_samples,))
        * gamma_b.sample((n_samples,))
    )
    with torch.no_grad():
        npt.assert_array_almost_equal(w_true.mean(0), w.mean(0), decimal=2)
        npt.assert_array_almost_equal(w_true.var(0), w.var(0), decimal=2)


class InvGamma(nn.Module):
    def __init__(self, alpha: Tensor, beta: Tensor) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

    def log_prob(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            y = invgamma.logpdf((x / self.beta).numpy(), self.alpha.numpy())
        return torch.from_numpy(y)


def test_half_cauchy_prior_kl_divergence():
    """
    This test checks that the exact kl divergence computations match with a 
    Monte Carlo estimate for the kl divergence between the variational and prior distributions
    """
    d = 2
    prior = SVIHalfCauchyPrior(d, 1e-1)
    n_samples = 2000000
    # variational distributions
    distributions = {
        "s_a": {
            "Variational": LogNormal(prior.s_a.mu, prior.s_a.var().sqrt()),
            "Prior": Gamma(torch.tensor(0.5), prior.tau.pow(2)),
            "kl_exact": prior._kl_s_a(),
        },
        "wi": {
            "Variational": Normal(prior.w_tilde.mu, prior.w_tilde.var().sqrt()),
            "Prior": Normal(torch.zeros(d), torch.ones(d)),
            "kl_exact": prior._kl_w_tilde(),
        },
        "s_b": {
            "Variational": LogNormal(prior.s_b.mu, prior.s_b.var().sqrt()),
            "Prior": InvGamma(torch.tensor(0.5), torch.tensor(1.0)),
            "kl_exact": prior._kl_s_b(),
        },
        "gamma_a": {
            "Variational": LogNormal(prior.gamma_a.mu, prior.gamma_a.var().sqrt()),
            "Prior": Gamma(torch.ones(d) * 0.5, torch.ones(d)),
            "kl_exact": prior._kl_gamma_a(),
        },
        "gamma_b": {
            "Variational": LogNormal(prior.gamma_b.mu, prior.gamma_b.var().sqrt()),
            "Prior": InvGamma(torch.ones(d) * 0.5, torch.ones(d)),
            "kl_exact": prior._kl_gamma_b(),
        },
    }
    kl = 0.0
    for param in distributions.keys():
        z = distributions[param]["Variational"].sample((n_samples,))
        logqz = distributions[param]["Variational"].log_prob(z)
        logpz = distributions[param]["Prior"].log_prob(z)
        kl_tmp = (logqz - logpz).mean(0).sum()
        print(kl_tmp)
        with torch.no_grad():
            err_msg = "KL for {} failed".format(param)
            npt.assert_array_almost_equal(
                kl_tmp, distributions[param]["kl_exact"], err_msg=err_msg, decimal=2
            )
        kl += kl_tmp

    with torch.no_grad():
        npt.assert_array_almost_equal(kl, prior.kl_divergence(), decimal=1)


def test_update_sparse_index():
    """
    Updates the sparse index
    """
    d = 4
    prior = SVIHalfCauchyPrior(d, 1e-5)
    sp_ind = torch.tensor([1, 0, 1, 0]).bool()
    prior._propogate_sparse_index(sp_ind)
    w = prior.get_reparam_weights(10)
    assert w.shape[0] == 10
    assert w.shape[-1] == 2
