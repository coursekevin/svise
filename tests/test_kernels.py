import svise.kernels as kernels
import torch
from svise.variationalsparsebayes.sparse_glm import inverse_softplus
import gpytorch
import pytest
from numpy import testing

torch.set_default_dtype(torch.float64)


def test_difference():
    x = torch.randn(3, 1)
    y = torch.randn(3, 1)
    res = kernels.difference(x, y)
    res_classic = x.unsqueeze(-2) - y.unsqueeze(-3)
    with torch.no_grad():
        testing.assert_array_equal(res, res_classic)


def test_identity_warp():
    x = torch.randn(20, 3)
    k = kernels.IdentityWarp()
    x.requires_grad = True
    f, df = k(x, return_grad=True)
    df_ag = torch.autograd.grad(f.sum(), x, retain_graph=True)[0]
    with torch.no_grad():
        testing.assert_array_almost_equal(df, df_ag)


def test_kumaraswamy_warping():
    x = torch.randn(20, 3)
    k = kernels.KumaraswamyWarping((x.min(), x.max()))
    k.rawalpha = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(2.1)))
    k.rawbeta = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(0.5)))
    x.requires_grad = True
    f, df = k(x, return_grad=True)
    # ag test
    df_ag = torch.autograd.grad(f.sum(), x, retain_graph=True)[0]
    with torch.no_grad():
        testing.assert_array_almost_equal(df, df_ag)
        testing.assert_array_almost_equal(x, k.inverse(f))
    # inverse test


def test_matern52():
    gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=5 / 2)
    kernel = kernels.Matern52()
    kernel.rawsigf = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(1.0)))
    kernel.rawlen = torch.nn.parameter.Parameter(
        inverse_softplus(gpytorch_kernel.lengthscale)
    )
    t1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    t2 = torch.linspace(1, 10, 10).reshape(-1, 1)

    with torch.no_grad():
        testing.assert_array_equal(kernel(t1, t2).shape, (10, 10))
        testing.assert_array_almost_equal(
            kernel(t1, t2), gpytorch_kernel(t1, t2).evaluate()
        )


def test_matern12():
    gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=1 / 2)
    kernel = kernels.Matern12()
    kernel.rawsigf = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(1.0)))
    kernel.rawlen = torch.nn.parameter.Parameter(
        inverse_softplus(gpytorch_kernel.lengthscale)
    )
    t1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    t2 = torch.linspace(1, 10, 10).reshape(-1, 1)

    with torch.no_grad():
        testing.assert_array_equal(kernel(t1, t2).shape, (10, 10))
        testing.assert_array_almost_equal(
            kernel(t1, t2), gpytorch_kernel(t1, t2).evaluate()
        )


def test_matern12_dkdt():
    kernel = kernels.Matern12()
    with torch.no_grad():
        kernel.rawsigf = torch.nn.Parameter(torch.tensor(1.0))
        kernel.rawlen = torch.nn.Parameter(torch.tensor(1.0))
    t = torch.linspace(0, 10, 10, requires_grad=True).reshape(-1, 1)
    # note: why does torch.linspace(0, 1, 10) fail?
    tau = torch.linspace(0, 10, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern12_with_warping():
    input_warping = kernels.KumaraswamyWarping((torch.tensor(0), torch.tensor(10)))
    input_warping.rawalpha = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(2.1))
    )
    input_warping.rawbeta = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(0.5))
    )
    kernel = kernels.Matern12(input_warping=input_warping)
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern32():
    gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=3 / 2)
    kernel = kernels.Matern32()
    kernel.rawsigf = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(1.0)))
    kernel.rawlen = torch.nn.parameter.Parameter(
        inverse_softplus(gpytorch_kernel.lengthscale)
    )
    t1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    t2 = torch.linspace(1, 10, 10).reshape(-1, 1)

    with torch.no_grad():
        testing.assert_array_equal(kernel(t1, t2).shape, (10, 10))
        testing.assert_array_almost_equal(
            kernel(t1, t2), gpytorch_kernel(t1, t2).evaluate()
        )


def test_matern32_dkdt():
    kernel = kernels.Matern32()
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    # note: why does torch.linspace(0, 1, 10) fail?
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern32_with_warping():
    input_warping = kernels.KumaraswamyWarping((torch.tensor(0), torch.tensor(10)))
    input_warping.rawalpha = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(2.1))
    )
    input_warping.rawbeta = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(0.5))
    )
    kernel = kernels.Matern32(input_warping=input_warping)
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern52_dkdt():
    gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=5 / 2)
    kernel = kernels.Matern52()
    with torch.no_grad():
        kernel.logsigf = torch.nn.Parameter(torch.tensor(0.0))
        kernel.loglen = torch.nn.Parameter(torch.log(gpytorch_kernel.lengthscale))
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    # note: why does torch.linspace(0, 1, 10) fail?
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern52_with_warping():
    input_warping = kernels.KumaraswamyWarping((torch.tensor(0), torch.tensor(10)))
    input_warping.rawalpha = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(2.1))
    )
    input_warping.rawbeta = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(0.5))
    )
    kernel = kernels.Matern52(input_warping=input_warping)
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(len(tau)):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(dKdt_fast, dKdt_autograd)


def test_matern52wgrad():
    gpytorch_kernel = gpytorch.kernels.MaternKernel(nu=5 / 2)
    kernel = kernels.Matern52withGradients()
    kernel.rawsigf = torch.nn.parameter.Parameter(inverse_softplus(torch.tensor(1.0)))
    kernel.rawlen = torch.nn.parameter.Parameter(
        inverse_softplus(gpytorch_kernel.lengthscale)
    )
    t = torch.linspace(0, 1, 10).reshape(-1, 1)
    tau = torch.linspace(1, 10, 20, requires_grad=True).reshape(-1, 1)
    Ktaut = gpytorch_kernel(t, tau).evaluate().t()
    dKdtau = []
    for j in range(len(t)):
        dKdtau.append(
            torch.autograd.grad(Ktaut[:, j].sum(), tau, retain_graph=True)[0].squeeze()
        )
    dKttaudtau_ag = torch.stack(dKdtau, dim=-1).t()
    with torch.no_grad():
        testing.assert_array_equal(kernel(t, tau).shape, (10, 2 * len(tau)))
        k_custom = kernel(t, tau)[:, : len(tau)]
        dk_custom = kernel(t, tau)[:, len(tau) :]
        testing.assert_array_almost_equal(k_custom, Ktaut.t())
        testing.assert_array_almost_equal(dk_custom, dKttaudtau_ag)


def test_matern52wgrad_dkdt():
    kernel = kernels.Matern52withGradients()
    t = torch.linspace(0, 10, 10, requires_grad=True).reshape(-1, 1)
    # note: why does torch.linspace(0, 1, 10) fail?
    tau = torch.linspace(0, 10, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(K.shape[-1]):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(
            dKdt_fast[:, : len(tau)], dKdt_autograd[:, : len(tau)]
        )
        testing.assert_array_almost_equal(
            dKdt_fast[:, len(tau) :], dKdt_autograd[:, len(tau) :]
        )


def test_matern52wgrad_with_warping():
    input_warping = kernels.KumaraswamyWarping((torch.tensor(0), torch.tensor(10)))
    input_warping.rawalpha = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(2.1))
    )
    input_warping.rawbeta = torch.nn.parameter.Parameter(
        inverse_softplus(torch.tensor(0.5))
    )
    kernel = kernels.Matern52withGradients(input_warping=input_warping)
    t = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    tau = torch.linspace(0, 1, 20).reshape(-1, 1)
    K = kernel(t, tau)
    dKdt = []
    for j in range(K.shape[-1]):
        dKdt.append(
            torch.autograd.grad(K[:, j].sum(), t, retain_graph=True)[0].squeeze()
        )
    dKdt_autograd = torch.stack(dKdt, dim=-1)
    # dKdt_fast = kernel.dkdt(t, tau)
    _, dKdt_fast = kernel(t, tau, return_grad=True)
    with torch.no_grad():
        testing.assert_array_almost_equal(
            dKdt_fast[:, : len(tau)], dKdt_autograd[:, : len(tau)]
        )
        testing.assert_array_almost_equal(
            dKdt_fast[:, len(tau) :], dKdt_autograd[:, len(tau) :]
        )

def test_rbf():
    gpytorch_kernel = gpytorch.kernels.RBFKernel()
    kernel = kernels.SquaredExpKernel()
    kernel.loglen = torch.nn.Parameter(torch.log(gpytorch_kernel.lengthscale))
    kernel.logsigf = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
    t1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    t2 = torch.linspace(0, 1, 10).reshape(-1, 1)

    with torch.no_grad():
        testing.assert_array_equal(kernel(t1, t2).shape, (10, 10))
        testing.assert_array_almost_equal(
            kernel(t1, t2), gpytorch_kernel(t1, t2).evaluate()
        )


def test_rbf_dkdt():
    gpytorch_kernel = gpytorch.kernels.RBFKernel()
    kernel = kernels.SquaredExpKernel()
    kernel.loglen = torch.nn.Parameter(torch.log(gpytorch_kernel.lengthscale))
    kernel.logsigf = torch.nn.Parameter(torch.log(torch.tensor([1.0])))
    t1 = torch.linspace(0, 1, 10).reshape(-1, 1)
    t2 = torch.linspace(0, 1, 10).reshape(-1, 1)

    with torch.no_grad():
        testing.assert_array_equal(kernel(t1, t2).shape, (10, 10))
        testing.assert_array_almost_equal(
            kernel(t1, t2), gpytorch_kernel(t1, t2).evaluate()
        )
