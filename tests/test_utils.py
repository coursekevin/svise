import numpy.testing as npt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from svise.utils import *
import torch
import timeit

torch.set_default_dtype(torch.float64)


def init_skew_sym(d, v=None):
    """
    Initializes a symmetric positive definite matrix of size d.
    """
    triu_ind = torch.triu_indices(d, d, offset=+1)
    S = torch.zeros(d, d)
    if v is None:
        v = torch.randn(d * (d - 1) // 2)
    else:
        assert len(v) == d * (d - 1) // 2, "Supplied v is the incorrect size."
    S[triu_ind[0], triu_ind[1]] = -v.clone()
    S[triu_ind[1], triu_ind[0]] = v.clone()
    return S


def init_spd(d):
    """
    Initializes a symmetric positive definite matrix of size d.
    """
    tril_ind = torch.tril_indices(d, d)
    Q = torch.zeros(d, d)
    Q[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
    Q = Q @ Q.t() + torch.eye(d) * 1e-3
    return Q


def test_make_random_matrix():
    A = make_random_matrix(50, 27)
    assert torch.linalg.matrix_rank(A) == 27
    B = make_random_matrix(50, 27, random_seed=200)
    with torch.no_grad():
        npt.assert_array_equal(B, make_random_matrix(50, 27, random_seed=200))


def test_standardize_transform():
    x = torch.randn(100, 2) * torch.tensor([2.5, 10.0]) + torch.tensor([1.0, 20.0])
    tfm = StandardizeTransform(x)
    y = tfm(x)
    with torch.no_grad():
        npt.assert_array_almost_equal(y.mean(0), torch.zeros(2))
        npt.assert_array_almost_equal(y.var(0), torch.ones(2))
        npt.assert_array_almost_equal(tfm.inverse(y), x)


def test_multivariate_standardize_tfm():
    d = 5
    tril_ind = torch.tril_indices(d, d)
    Q = torch.zeros(d, d)
    Q[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
    Sigma = Q @ Q.t() + torch.eye(d) * 1e-3
    L = torch.linalg.cholesky(Sigma)
    x = torch.randn(1000, d) @ L.t() + torch.arange(d)
    tfm = MultivariateStandardizeTransform(x)
    y = tfm(x)
    with torch.no_grad():
        npt.assert_array_almost_equal(y.mean(0), torch.zeros(d))
        npt.assert_array_almost_equal(sample_covariance(y), torch.eye(d))
        npt.assert_array_almost_equal(tfm.inverse(y), x)


def test_bjorck():
    O, _ = bjorck(torch.randn(200, 5))
    with torch.no_grad():
        npt.assert_array_almost_equal(O.T @ O, torch.eye(5))
    O, _ = bjorck(torch.randn(32, 200, 5))
    with torch.no_grad():
        npt.assert_array_almost_equal(
            O.transpose(-2, -1) @ O, torch.eye(5).repeat(32, 1, 1)
        )


def test_fcnn():
    # check in and out sizes are as expected
    model = FCNN(input_size=2, hidden_size=50, output_size=3, num_layers=3)
    npt.assert_array_equal(model(torch.randn(20, 2)).shape, (20, 3))
    # check we can do better than a constant model
    x = torch.linspace(-1, 1, 100)
    noise = 1e-1
    y = torch.sin(x * 2 * math.pi) + x + (noise * torch.randn(x.shape))
    # linear model
    w = (x @ y) / (x @ x)
    model = FCNN(input_size=1, hidden_size=50, output_size=1, num_layers=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for i in range(500):
        optimizer.zero_grad()
        loss = (model(x.unsqueeze(-1)).squeeze() - y).pow(2).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    loss_fcnn = (model(x.unsqueeze(-1)).squeeze() - y).pow(2).mean().item()
    loss_linear = (w * x - y).pow(2).mean().item()
    assert loss_fcnn < loss_linear


def test_fcnn_shape():
    # check in and out sizes are as expected
    model = FCNN(input_size=2, hidden_size=50, output_size=3, num_layers=3)
    model.eval()
    # check we can do better than a constant model
    x = torch.randn(30, 10, 2)
    y1 = torch.stack([model(xi) for xi in x])
    y2 = model(x)
    with torch.no_grad():
        npt.assert_array_almost_equal(y1, y2)


def test_positive_transform():
    f = Positive(beta=10.0)
    x = torch.linspace(-100, 100, 10)
    x.requires_grad = True
    res, grad_res = f(x, return_grad=True)
    grad_ag = torch.autograd.grad(res.sum(), x)[0]
    with torch.no_grad():
        npt.assert_array_almost_equal(grad_res, grad_ag)


def test_glm():
    # check in and out sizes are as expected
    # check we can do better than a constant model
    x = torch.linspace(-1, 1, 100)
    noise = 1e-1
    y = torch.sin(x * 2 * math.pi) + x + (noise * torch.randn(x.shape))
    # linear model
    w = (x @ y) / (x @ x)
    model = GLM(output_size=1, a=-1, b=1, n_tau=50)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for i in range(500):
        optimizer.zero_grad()
        loss = (model(x.unsqueeze(-1)).squeeze() - y).pow(2).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    loss_fcnn = (model(x.unsqueeze(-1)).squeeze() - y).pow(2).mean().item()
    loss_linear = (w * x - y).pow(2).mean().item()
    assert loss_fcnn < loss_linear


def test_multioutput_gradient():
    t = torch.linspace(-10, 10, 100, requires_grad=True)
    f = torch.stack([torch.sin(t), torch.cos(t * 2), t.pow(2)], dim=-1)
    dfdt = torch.stack([torch.cos(t), -2 * torch.sin(t * 2), 2 * t], dim=-1)
    df_ag = multioutput_gradient(f, t, vmap=False)
    # didn't end up using vmap impl, removing for now
    df_ag_vmap = multioutput_gradient(f, t, vmap=False) 
    with torch.no_grad():
        npt.assert_array_equal(df_ag, dfdt)
        npt.assert_array_equal(df_ag_vmap, dfdt)


def test_sample_covariance():
    x = [torch.randn(100, 3) for i in range(10)]
    C_np = torch.stack([torch.from_numpy(np.cov(xi.t())) for xi in x])
    C = sample_covariance(torch.stack(x))
    with torch.no_grad():
        npt.assert_array_almost_equal(C, C_np)


def test_euler_maruyama():
    theta = 1e-1
    sigma = 0.5

    # defining ornstein-uhlenbeck process
    def f(t, x):
        return -theta * x

    def L(t, x):
        return sigma * torch.ones_like(x).unsqueeze(-1)

    # integrating with eulermaruyama
    Q = torch.eye(1)
    x0 = 5.0
    nsamples = 2000
    sde = EulerMaruyama(f, L, Q, (0, 25), x0 * torch.ones(nsamples, 1), 1e-2)
    C_sample = sample_covariance(sde.x.squeeze())

    t = sde.t
    # computing the true mean
    mean_true = x0 * torch.exp(-theta * t)
    diff = t.unsqueeze(-1) - t.unsqueeze(0)
    add = t.unsqueeze(-1) + t.unsqueeze(0)
    # computing the true covariance
    C_true = (
        sigma ** 2
        / (2 * theta)
        * (torch.exp(-theta * diff.abs()) - torch.exp(-theta * add))
    )
    with torch.no_grad():
        npt.assert_array_almost_equal(mean_true, sde.mean.squeeze(), decimal=1)
        npt.assert_array_almost_equal(C_true, C_sample, decimal=1)
        npt.assert_array_almost_equal(C_true.diag(), sde.var.squeeze(), decimal=1)


def test_solve_least_squares():
    n = 100
    m = 50
    d = 4
    A = torch.randn(n, m)
    B = torch.randn(n, d)
    C = torch.linalg.cholesky(A.t() @ A)
    naive_W = torch.cholesky_solve(A.t() @ B, C)
    stable_W = solve_least_squares(A, B)
    with torch.no_grad():
        npt.assert_array_almost_equal(naive_W, stable_W)


def test_solve_lyapunov_spectral():
    d = 10
    tril_ind = torch.tril_indices(d, d)
    # setting up some spd matrices
    Q = torch.zeros(d, d)
    Q[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
    Q = Q @ Q.t()
    K = torch.zeros(d, d)
    K[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
    K = K @ K.t() + torch.eye(d) * 1e-3
    # solving the lyapunov equation using eig
    D, R = torch.linalg.eigh(K)
    st_time = timeit.default_timer()
    X = solve_lyapunov_spectral(D, R, Q)
    fast_time = timeit.default_timer() - st_time
    Q_solve = K @ X.t() + X @ K.t()
    with torch.no_grad():
        npt.assert_array_almost_equal(Q_solve, Q, decimal=4)
    # check it's actually fast
    st_time = timeit.default_timer()
    # naive solve
    I = torch.eye(d)
    LHS = torch.kron(I, K) + torch.kron(K, I)
    vecX = torch.linalg.solve(LHS, Q.t().ravel())
    slow_time = timeit.default_timer() - st_time
    assert slow_time > fast_time


def test_solve_lyapunov_spectral_batch():
    d = 10
    m = 3
    tril_ind = torch.tril_indices(d, d)
    # setting up some spd matrices
    Q_in = []
    K_in = []
    for j in range(m):
        Q = torch.zeros(d, d)
        Q[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
        Q = Q @ Q.t()
        Q_in.append(Q)
        K = torch.zeros(d, d)
        K[tril_ind[0], tril_ind[1]] = torch.randn(d * (d + 1) // 2)
        K = K @ K.t() + torch.eye(d) * 1e-3
        K_in.append(K)
    Q_in = torch.stack(Q_in)
    K_in = torch.stack(K_in)
    # solving the lyapunov equation using eig
    D, R = torch.linalg.eigh(K_in)
    st_time = timeit.default_timer()
    X = solve_lyapunov_spectral(D, R, Q_in)
    fast_time = timeit.default_timer() - st_time
    Q_solve = K_in @ X.transpose(-2, -1) + X @ K_in.transpose(-2, -1)
    with torch.no_grad():
        npt.assert_array_almost_equal(Q_solve, Q_in, decimal=4)


def test_solve_lyapunov_spectral_double_batch():
    d = 10
    m = 3
    tril_ind = torch.tril_indices(d, d)
    # setting up some spd matrices
    K_in = []
    for j in range(m):
        K = init_spd(d) + torch.eye(d) * 1e-3
        K_in.append(K)
    Q_in = torch.stack([torch.stack([init_spd(d) for i in range(3)]) for i in range(5)])
    K_in = torch.stack(K_in)
    # solving the lyapunov equation using eig
    D, R = torch.linalg.eigh(K_in)
    st_time = timeit.default_timer()
    X = solve_lyapunov_spectral(D, R, Q_in)
    fast_time = timeit.default_timer() - st_time
    Q_solve = K_in @ X.transpose(-2, -1) + X @ K_in.transpose(-2, -1)
    with torch.no_grad():
        npt.assert_array_almost_equal(
            Q_solve - Q_in, torch.zeros(5, 3, d, d), decimal=4
        )


def test_solve_lyapunov_diag():
    d = 50
    # setting up some matrices
    Q = torch.randn(d,).diag()
    K = torch.randn(d,).diag().pow(2) + 1e-3 * torch.eye(d)
    # solving the lyapunov equation using diag
    st_time = timeit.default_timer()
    X = torch.diag(solve_lyapunov_diag(K.diag(), Q.diag()))
    fast_time = timeit.default_timer() - st_time
    Q_solve = K @ X.t() + X @ K.t()
    with torch.no_grad():
        npt.assert_array_almost_equal(Q_solve, Q, decimal=4)
    # check it's actually fast
    st_time = timeit.default_timer()
    # naive solve
    I = torch.eye(d)
    LHS = torch.kron(I, K) + torch.kron(K, I)
    vecX = torch.linalg.solve(LHS, Q.t().ravel())
    slow_time = timeit.default_timer() - st_time
    assert slow_time > fast_time


def test_solve_lyapunov_diagonal_double_batch():
    d = 10
    m = 3
    tril_ind = torch.tril_indices(d, d)
    # setting up some spd matrices
    Q_in = torch.randn(5, d).pow(2)
    K_in = torch.randn(m, d).pow(2)
    # solving the lyapunov equation using eig
    st_time = timeit.default_timer()
    X = solve_lyapunov_diag(K_in, Q_in).diag_embed()
    fast_time = timeit.default_timer() - st_time
    Q_solve = K_in.diag_embed() @ X.transpose(-2, -1) + X @ K_in.diag_embed().transpose(
        -2, -1
    )
    with torch.no_grad():
        npt.assert_array_almost_equal(
            Q_solve - Q_in.diag_embed().unsqueeze(1), torch.zeros(5, 3, d, d), decimal=4
        )


def test_isotropic_gaussian_loglike():
    from torch.distributions.multivariate_normal import MultivariateNormal as MVN

    d = 8
    bs = 4
    ns = 2

    x = torch.randn(bs, d)
    mu = torch.randn(ns, bs, d)
    var = torch.randn(d,).pow(2)
    log_like_fast = mean_diagonal_gaussian_loglikelihood(x, mu, var)
    log_like_manual = 0.0
    for i in range(ns):
        for j in range(bs):
            covar = torch.eye(d) * var
            mvn = MultivariateNormal(mu[i, j], covariance_matrix=covar)
            log_like_manual += mvn.log_prob(x[j]) / (bs * ns)
    with torch.no_grad():
        npt.assert_array_almost_equal(log_like_fast, log_like_manual, decimal=4)


def test_matrix_logm():
    d = 5
    A = torch.randn(d, d)
    expA = torch.matrix_exp(A)
    # checking matrix log is correct
    with torch.no_grad():
        npt.assert_array_almost_equal(matrix_log(expA), A)
    # checking gradient
    bs = 10
    t = torch.linspace(0, 1, bs, requires_grad=True)
    tril_ind = torch.tril_indices(d, d, offset=-1)
    Q = torch.zeros(bs, d, d)
    v = torch.randn(d * (d - 1) // 2)

    def f(t):
        u = (
            v.unsqueeze(0)
            + v.unsqueeze(0) * t.unsqueeze(-1)
            + v.unsqueeze(0) * t.unsqueeze(-1).pow(2)
        )
        Q[:, tril_ind[0], tril_ind[1]] = u
        Q[:, tril_ind[1], tril_ind[0]] = -u
        return matrix_log(torch.matrix_exp(Q))

    f_fd = finite_difference(f, t, 1e-9)
    f_ag = multioutput_gradient(f(t).reshape(bs, -1), t, vmap=False).reshape(bs, d, d)

    # checking gradient is correct
    with torch.no_grad():
        npt.assert_array_almost_equal(f_fd, f_ag, decimal=2)


def test_skew_sym_duplication_matrix():
    d = 4
    v = torch.arange(d * (d - 1) // 2) + 1.0
    S = init_skew_sym(d, v=v)
    D_n = skew_sym_duplication_matrix(d)
    npt.assert_array_equal(S.t().ravel(), D_n @ v)


def test_skew_sym_matrix_exp():
    d = 5
    init_skew_sym(d)
    orth_matrix_exp = SkewSymMatrixExp(d)
    ns = 100
    v = [torch.randn(d * (d - 1) // 2) for _ in range(ns)]
    S = [init_skew_sym(d, v=v[i]) for i in range(ns)]
    with torch.no_grad():
        npt.assert_array_almost_equal(torch.matrix_exp(S[0]), orth_matrix_exp(v[0]))
        npt.assert_array_almost_equal(
            torch.matrix_exp(torch.stack(S)), orth_matrix_exp(torch.stack(v))
        )


def test_skew_sym_matrix_exp_grad():
    d = 5
    v = torch.randn(d * (d - 1) // 2)
    v.requires_grad = True
    S = init_skew_sym(d, v=v)
    orth_matrix_exp = SkewSymMatrixExp(d)
    mexp = torch.matrix_exp(S)
    grad_expS_slow = []
    # single input
    for j in range(d ** 2):
        grad_expS_slow.append(
            torch.autograd.grad(mexp.t().flatten()[j], v, retain_graph=True)[0]
        )
    grad_expS_slow = torch.stack(grad_expS_slow)
    _, grad_oexp = orth_matrix_exp(v, return_grad=True)
    with torch.no_grad():
        npt.assert_array_almost_equal(grad_oexp, grad_expS_slow)
    # multiple inputs
    v = [torch.randn(d * (d - 1) // 2) for _ in range(8)]
    for vi in v:
        vi.requires_grad = True
    S = [init_skew_sym(d, v=v[i]) for i in range(len(v))]
    mexp_stack = [torch.matrix_exp(Si) for Si in S]
    grad_expS_slow = []
    for i in range(len(mexp_stack)):
        grad_expS_slow_tmp = []
        for j in range(d ** 2):
            grad_expS_slow_tmp.append(
                torch.autograd.grad(
                    mexp_stack[i].t().flatten()[j], v[i], retain_graph=True
                )[0]
            )
        grad_expS_slow.append(torch.stack(grad_expS_slow_tmp))
    grad_expS_slow = torch.stack(grad_expS_slow)
    _, grad_oexp = orth_matrix_exp(torch.stack(v), return_grad=True)
    with torch.no_grad():
        npt.assert_array_almost_equal(grad_oexp, grad_expS_slow)


def test_skew_sym_grad_pipeline():
    d = 5
    bs = 1000
    fast_expm = SkewSymMatrixExp(d)
    t = torch.linspace(-10, 10, bs, requires_grad=True)
    f = torch.stack(
        [1.1 + torch.sin((t - j) * 2 * math.pi) for j in range(d * (d - 1) // 2)],
        dim=-1,
    )
    dfdt = torch.stack(
        [
            2 * math.pi * torch.cos((t - j) * 2 * math.pi)
            for j in range(d * (d - 1) // 2)
        ],
        dim=-1,
    )
    dfdt_ag = multioutput_gradient(f, t, vmap=False)
    # check test problem is correct
    with torch.no_grad():
        npt.assert_array_almost_equal(dfdt, dfdt_ag)

    # construct skew symmetric matrix
    tril_ind = torch.triu_indices(d, d, offset=+1)
    S = torch.zeros(bs, d, d)
    S[:, tril_ind[1], tril_ind[0]] = f.clone()
    S[:, tril_ind[0], tril_ind[1]] = -f.clone()

    # autograd
    st_time = timeit.default_timer()
    expm = torch.matrix_exp(S)
    dexpmdt_ag = multioutput_gradient(
        expm.transpose(-2, -1).reshape(bs, -1), t, vmap=False
    )
    ag_time = timeit.default_timer() - st_time

    # manual
    st_time = timeit.default_timer()
    _, grad_expm = fast_expm(f, return_grad=True)
    dexpmdt_fast = (grad_expm @ dfdt.unsqueeze(-1)).squeeze()
    manual_time = timeit.default_timer() - st_time
    with torch.no_grad():
        npt.assert_array_almost_equal(dexpmdt_fast, dexpmdt_ag)

    assert ag_time > manual_time


class SkewSymMatrixExpLegacy(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.register_buffer("dupl_mat", skew_sym_duplication_matrix(d) + 0j)
        self.register_buffer("ind", torch.triu_indices(d, d, offset=+1))
        self.register_buffer("d_ind", torch.arange(d))

    def forward(self, S_vec, return_grad=False):
        S = torch.zeros((*S_vec.shape[:-1], self.d, self.d))
        S[..., self.ind[0], self.ind[1]] = -S_vec
        S[..., self.ind[1], self.ind[0]] = S_vec
        D, V = torch.linalg.eigh(1.0j * S)  # j * S is Hermitian
        D = D / 1.0j
        exp_D = torch.exp(D)
        V_trns_conj = V.transpose(-2, -1).conj()
        exp_S = (V * exp_D.unsqueeze(-2) @ V_trns_conj).real
        if not return_grad:
            return exp_S
        else:
            # https://www.janmagnus.nl/wips/expo-23.pdf
            # todo: can we write Delta in terms of a kronecker product?
            Delta = torch.zeros_like(V)
            diff = (exp_D[..., self.ind[0]] - exp_D[..., self.ind[1]]) / (
                D[..., self.ind[0]] - D[..., self.ind[1]]
            )
            Delta[..., self.ind[0], self.ind[1]] = diff
            Delta[..., self.ind[1], self.ind[0]] = diff
            Delta[..., self.d_ind, self.d_ind] = exp_D
            if S.dim() == 2:
                R = torch.kron(V.conj(), V)
                R_inv_dupl = (
                    torch.kron(V.transpose(-2, -1), V_trns_conj) @ self.dupl_mat
                )
            elif S.dim() == 3:
                # batch wise kronecker product
                R = torch.einsum("ijk,ilm->ijlkm", V.conj(), V).view(
                    -1, self.d ** 2, self.d ** 2
                )
                R_inv_dupl = (
                    torch.einsum(
                        "ijk,ilm->ijlkm", V.transpose(-2, -1), V_trns_conj
                    ).view(-1, self.d ** 2, self.d ** 2)
                    @ self.dupl_mat
                )
            else:
                raise ValueError("S must be 2 or 3 dimensional")
            grad_exp_S = (
                R * Delta.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) @ (R_inv_dupl)
            )
            return exp_S, grad_exp_S.real


def test_skew_matrix_exp_new():
    d = 5
    v = torch.randn(100, d * (d - 1) // 2, requires_grad=True)
    fast = SkewSymMatrixExp(d)
    legacy = SkewSymMatrixExpLegacy(d)
    O_fast, grad_fast = fast(v, return_grad=True)
    O_legacy, grad_legacy = legacy(v, return_grad=True)
    with torch.no_grad():
        npt.assert_array_almost_equal(O_fast, O_legacy)
        npt.assert_array_almost_equal(grad_fast, grad_legacy)


def test_fast_skew_sym_matrix_exp():
    d = 5
    v = torch.randn(100, d * (d - 1) // 2, requires_grad=True)
    module = SkewSymMatrixExp(d)
    f = lambda x: fast_skew_sym_matrix_exp(x, module.ind, module.d_ind, module.dupl_mat)
    assert torch.autograd.gradcheck(f, v)


def test_stdev_scale_transform():
    x = torch.randn(1000, 5)
    tfm = StdevScaleTransform(x)
    with torch.no_grad():
        npt.assert_array_almost_equal(tfm(x).std(0), torch.ones(5))
        npt.assert_array_almost_equal(tfm.inverse(tfm(x)), x)
        npt.assert_array_almost_equal(tfm(x) * tfm.scale, x)


def test_identity_scale_transform():
    x = torch.randn(1000, 5)
    tfm = IdentityScaleTransform(x.shape[-1])
    with torch.no_grad():
        npt.assert_array_almost_equal(tfm(x), x)
        npt.assert_array_almost_equal(tfm.inverse(tfm(x)), x)
        npt.assert_array_almost_equal(tfm(x) * tfm.scale, x)
