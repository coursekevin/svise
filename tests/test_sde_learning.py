import time

import numpy.testing as npt
import pytest
import torch
from torch import nn
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal

from svise.sde_learning import *
from svise.utils import *

torch.set_default_dtype(torch.float64)


def init_spd(d):
    """
    Initializes a symmetric positive definite matrix of size d.
    """
    tril_ind = torch.tril_indices(d, d)
    Q = torch.zeros(d, d)
    Q[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
    Q = Q @ Q.t() + torch.eye(d) * 1e-3
    return Q


def test_finite_difference():
    f = lambda t: torch.stack([torch.sin(t), torch.cos(t)], dim=-1)
    g = lambda t: torch.stack([torch.cos(t), -torch.sin(t)], dim=-1)
    t = torch.linspace(0, 1, 10, requires_grad=True)
    with torch.no_grad():
        npt.assert_array_almost_equal(g(t), finite_difference(f, t, 1e-6), decimal=4)


@pytest.fixture
def sparse_diag_diffusion_prior():
    d = 4
    Q_diag = torch.randn(d).pow_(2)
    n_reparam_samples = 32
    model = SparseDiagonalDiffusionPrior(d, Q_diag, n_reparam_samples, 1e-5)
    model.eval()
    return model


def test_sparse_diag_resample(sparse_diag_diffusion_prior):
    model = sparse_diag_diffusion_prior
    Sigma_diag = model.Sigma_diag
    # check sigma diag
    shape = (model.n_reparam_samples, model.d)
    assert Sigma_diag.shape == shape
    with torch.no_grad():
        npt.assert_array_equal(
            torch.zeros(shape), (model.Sigma_diag - Sigma_diag).abs()
        )
    model.resample_weights()
    with torch.no_grad():
        npt.assert_array_less(torch.zeros(shape), (model.Sigma_diag - Sigma_diag).abs())


def test_sparse_diag_process_noise(sparse_diag_diffusion_prior):
    model = sparse_diag_diffusion_prior
    Sigma_diag = model.Sigma_diag
    Q = model.Q
    Sigma = Sigma_diag.diag_embed()
    with torch.no_grad():
        npt.assert_array_almost_equal(
            Sigma @ Q @ Sigma.transpose(-1, -2), model.process_noise
        )


def test_sparse_diag_kl_divergence(sparse_diag_diffusion_prior):
    model = sparse_diag_diffusion_prior
    with torch.no_grad():
        assert model.kl_divergence == model.prior.kl_divergence()


def test_sparse_diag_solve_linear(sparse_diag_diffusion_prior):
    model = sparse_diag_diffusion_prior
    r = torch.randn(32, 100, model.d)
    chol_r = torch.cholesky_solve(
        r.transpose(-2, -1), torch.linalg.cholesky(model.process_noise)
    ).transpose(-2, -1)
    with torch.no_grad():
        npt.assert_array_almost_equal(model.solve_linear(r), chol_r)


@pytest.fixture
def spectral_marginal_sde_instance():
    d = 4
    # initializing a random diffusion matrix
    Q_diag = torch.randn(d).pow(2)
    dmodel = SparseDiagonalDiffusionPrior(d, Q_diag, 32, 1e-5)
    # Q = init_spd(2)
    # Sigma = torch.randn(4, 2)
    # dmodel = ConstantDiffusionPrior(d, Q, Sigma)
    # adding dummy n_reparam_samples for some tests
    dmodel.n_reparam_samples = 32
    model = SpectralMarginalSDE(
        d=d, t_span=(0.0, 1.0), diffusion_prior=dmodel, n_tau=10
    )
    model.eval()
    return model


def test_orthonorm_matrix(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    t = torch.linspace(0, 1, 10, requires_grad=True)
    param_method = [
        "exp",
    ]  # "cayley" is deprecated
    for pm in param_method:
        U, dUdt = model.orthogonal(t, return_grad=True)
        # check orthonormality
        with torch.no_grad():
            npt.assert_array_almost_equal(
                U @ U.transpose(-2, -1), torch.eye(model.d).repeat(10, 1, 1)
            )
        # check gradient computations
        dUdt_fd = finite_difference(
            lambda t: model.orthogonal(t).reshape(10, -1),
            t,
            1e-7,
        ).reshape(10, model.d, model.d)
        with torch.no_grad():
            npt.assert_array_almost_equal(dUdt, dUdt_fd, decimal=6)


def test_covariance_time_derivative(spectral_marginal_sde_instance):
    bs = 10
    model = spectral_marginal_sde_instance
    t = torch.linspace(0, 1, bs, requires_grad=True)
    dKdt_fd = finite_difference(model.K, t, 1e-6)
    start_time = time.time()
    # compute using autograd
    K_ag = model.K(t)
    dKdt_ag = multioutput_gradient(K_ag.reshape(bs, -1), t, vmap=False).reshape(
        K_ag.shape
    )
    t_autograd = time.time() - start_time
    # compute using intermediate quantities
    start_time = time.time()
    U, dUdt = model.orthogonal(t, return_grad=True)
    D = model.eigenvals(t)
    K = (U * D.unsqueeze(-2)) @ U.transpose(-2, -1)
    dDdt = multioutput_gradient(D, t, vmap=False)
    dKdt_fast = model._dKdt_fast(U, dUdt, D, dDdt)
    t_fast = time.time() - start_time
    with torch.no_grad():
        npt.assert_array_almost_equal(dKdt_ag, dKdt_fast, decimal=4)
        npt.assert_array_almost_equal(dKdt_fd, dKdt_fast, decimal=4)
    assert t_autograd > t_fast


def test_generate_samples(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    t = torch.linspace(0, 1, 10, requires_grad=True)
    n_samples = 100000
    z_s = model.generate_samples(t, n_samples)
    assert z_s.shape == (n_samples, 10, model.d)
    with torch.no_grad():
        npt.assert_array_almost_equal(model.mean(t), z_s.mean(0), decimal=1)
        npt.assert_array_almost_equal(
            model.K(t), sample_covariance(z_s.transpose(0, 1)), decimal=1
        )
    _, U, dUdt, D, dDdt = model.generate_samples(
        t, n_samples, return_intermediates=True
    )
    with torch.no_grad():
        npt.assert_array_almost_equal(
            (U * D.unsqueeze(-2) @ U.transpose(-2, -1)), model.K(t), decimal=4
        )


def test_sde_drift_shape(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    t = torch.linspace(0, 1, 10, requires_grad=True)
    n_reparam = model.diffusion_prior.n_reparam_samples
    z_s = model.generate_samples(t, n_reparam)
    drift = model.drift(t, z_s)
    assert drift.shape == (n_reparam, 10, model.d)


def test_sde_residual(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    t = torch.linspace(0, 1, 10, requires_grad=True)
    n_reparam = model.diffusion_prior.n_reparam_samples
    z = model.generate_samples(t, n_reparam)
    # this should actually have an additional batch dimension
    f = model.drift(t, z) + torch.randn(n_reparam, 10, model.d) * 1e-2
    r = model.unweighted_residual_loss(model.drift(t, z), f)
    # checking shape
    assert r.shape == (n_reparam, 10)
    # checking residual loss
    d = f - model.drift(t, z)
    p_noise = model.diffusion_prior.process_noise
    if p_noise.dim() == 3:
        r_linalg_solve = (
            d * torch.linalg.solve(p_noise, d.transpose(-2, -1)).transpose(-2, -1)
        ).sum(-1)
    else:
        U, S, Vh = torch.linalg.svd(p_noise, full_matrices=False)
        solve = (d @ (U * (S / (S.pow(2) + 1e-12)))) @ Vh.T
        r_linalg_solve = torch.einsum("ijk,ijk->ij", solve, d)
    with torch.no_grad():
        npt.assert_array_almost_equal(r, r_linalg_solve, decimal=2)


def test_forward_shape(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    n_samples = model.diffusion_prior.n_reparam_samples
    n_times = 10
    t = torch.linspace(0, 1, n_times, requires_grad=True)
    func = lambda t, z: torch.randn(n_samples, len(t), model.d)
    assert model(t, func, n_samples).shape == (n_times,)
    # test maximum likelhood version
    func = lambda t, z: torch.randn(n_times, model.d)
    assert model(t, func, n_samples).shape == (n_times,)


def test_autonomous_drift_fcnn(spectral_marginal_sde_instance):
    model = spectral_marginal_sde_instance
    f = AutonomousDriftFCNN(model.d, 200, 3)
    n_samples = 64
    n_times = 10
    t = torch.linspace(0, 1, n_times, requires_grad=True)
    z = model.generate_samples(t, n_samples)
    fz = f(t, z)
    assert fz.shape == (n_samples, n_times, model.d)
    assert f.kl_divergence() == 0.0


def test_meanglm():
    d = 3
    t = torch.linspace(0.0, 1.0, 10, requires_grad=True)
    f = MeanGLM(d, (0, 1), 10)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    dmdt_ag = multioutput_gradient(m, t, vmap=False)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
    m2 = f(t.unsqueeze(-1), return_grad=False)
    with torch.no_grad():
        npt.assert_array_equal(m, m2)


def test_strictly_positive_glm():
    d = 3
    t = torch.linspace(-100, 100, 10, requires_grad=True)
    f = StrictlyPositiveGLM(d, (-100, 100), 10)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    dmdt_ag = multioutput_gradient(m, t, vmap=False)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
        npt.assert_array_less(torch.zeros(10, d), m)
    m2 = f(t.unsqueeze(-1), return_grad=False)
    with torch.no_grad():
        npt.assert_array_equal(m, m2)


def test_orthogonal_glm():
    d = 5
    t = torch.linspace(-10.0, 10, 100, requires_grad=True)
    f = OrthogonalGLM(d, (-10.0, 10.0), 7)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    # check output is orthogonal
    O = m @ m.transpose(-2, -1)
    with torch.no_grad():
        npt.assert_array_almost_equal(O, torch.eye(d).repeat(100, 1, 1))
    # todo: test that grads are being computed correctly
    dmdt_ag = multioutput_gradient(m.reshape(100, -1), t, vmap=False).reshape(100, d, d)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
    # check no grad version is the same
    m2 = f(t.unsqueeze(-1))
    with torch.no_grad():
        npt.assert_array_almost_equal(m, m2)


def test_meanfcnn():
    d = 3
    t = torch.linspace(0.0, 1.0, 10, requires_grad=True)
    f = MeanFCNN(d, (0, 1), 50, 3)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    dmdt_ag = multioutput_gradient(m, t, vmap=False)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
    m2 = f(t.unsqueeze(-1), return_grad=False)
    with torch.no_grad():
        npt.assert_array_equal(m, m2)


def test_strictly_positive_fcnn():
    d = 3
    t = torch.linspace(-100, 100, 10, requires_grad=True)
    f = StrictlyPositiveFCNN(d, (0, 1), 50, 3)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    dmdt_ag = multioutput_gradient(m, t, vmap=False)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
        npt.assert_array_less(torch.zeros(10, d), m)
    m2 = f(t.unsqueeze(-1), return_grad=False)
    with torch.no_grad():
        npt.assert_array_equal(m, m2)


def test_orthogonal_fcnn():
    d = 5
    t = torch.linspace(-10.0, 10, 100, requires_grad=True)
    f = OrthogonalFCNN(d, (-10.0, 10.0), 50, 3)
    m, dmdt = f(t.unsqueeze(-1), return_grad=True)
    # check output is orthogonal
    O = m @ m.transpose(-2, -1)
    with torch.no_grad():
        npt.assert_array_almost_equal(O, torch.eye(d).repeat(100, 1, 1))
    # todo: test that grads are being computed correctly
    dmdt_ag = multioutput_gradient(m.reshape(100, -1), t, vmap=False).reshape(100, d, d)
    with torch.no_grad():
        npt.assert_array_almost_equal(dmdt, dmdt_ag)
    # check no grad version is the same
    m2 = f(t.unsqueeze(-1))
    with torch.no_grad():
        npt.assert_array_almost_equal(m, m2)


def test_driftfcnn():
    dim = 3
    layer_description = [50, 50, 50]
    nonlinearity = nn.Tanh()
    drift = DriftFCNN(dim, layer_description, nonlinearity)
    x = torch.randn(10, dim)
    t = torch.randn(10)
    assert drift(t, x).shape == (10, dim)
    x = torch.randn(20, 10, dim)
    assert drift(t, x).shape == (20, 10, dim)
    print("Starting", end="\n\n")
    assert 2 * (len(layer_description) + 1) == len(list(drift.named_parameters()))
    assert drift.kl_divergence() == 0.0
    assert torch.all(drift.forward(t, x) == drift.drift(t, x))
    # check it method has been implemented (even though it won't do anything for this class)
    drift.resample_weights()


def test_NeuralSDE():
    n_meas = 5
    dim = 3
    t_span = (0, 20)
    nsde_args = dict(
        d=dim,
        t_span=t_span,
        n_reparam_samples=10,
        G=torch.randn(n_meas, dim),
        drift_layer_description=[50, 50, 50],
        nonlinearity=nn.Tanh(),
        measurement_noise=torch.randn(n_meas).pow(2),
        tau=1e-5,
        train_t=torch.linspace(*t_span, 100),
        train_x=torch.randn(100, n_meas),
        n_quad=128,
        quad_percent=0.8,
        n_tau=200,
    )
    sde = NeuralSDE(**nsde_args) # type:ignore
    # check that loss can be computed and that backwards pass works
    loss = sde.elbo(nsde_args["train_t"], nsde_args["train_x"], beta=1.0, N = 100) # type: ignore
    loss.backward()
