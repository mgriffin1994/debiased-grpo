"""Numerical verification of the baseline variance + cross-sample covariance
scalings quoted in ``notes/derivation.md`` and the README.

All quantities are **conditional on a fixed prompt** — i.e. the group rewards are
i.i.d. with within-prompt variance σ², so between-prompt difficulty is removed.
Group size G; the independent baseline averages M reference draws. The advantage
for rollout i under each baseline is ``A_i = r_i − b_i``:

  - mean-with-self (MWS): ``b = (1/G) Σ_k r_k``      (includes r_i)
  - leave-one-out (LOO):  ``b_i = (1/(G−1)) Σ_{k≠i} r_k``
  - independent:          ``b = (1/M) Σ_m ρ_m``,  ρ ⟂ r  (shared across the group)

Closed forms verified here (σ² = Var(r|x)):

  Var(A_i)            MWS = σ²(G−1)/G   LOO = σ²·G/(G−1)   indep = σ²(1+1/M)
  Cov(A_i,A_j) i≠j    MWS = −σ²/G       LOO = −G·σ²/(G−1)² indep = +σ²/M
  Var(LOO)/Var(MWS)   = (G/(G−1))²
  Var(indep)/Var(MWS) = (1+1/M)·G/(G−1)
"""
import torch


def _advantages_mws(r):
    return r - r.mean(dim=1, keepdim=True)


def _advantages_loo(r):
    G = r.shape[1]
    return r - (r.sum(dim=1, keepdim=True) - r) / (G - 1)


def _advantages_independent(r, ref):
    return r - ref.mean(dim=1, keepdim=True)  # shared baseline, ⟂ r


def _var(A):
    return A.var(unbiased=False).item()


def _mean_cross_cov(A):
    """Average Cov(A_i, A_j) over i≠j, estimated across the N group draws."""
    Ac = A - A.mean(dim=0, keepdim=True)
    C = (Ac.t() @ Ac) / A.shape[0]            # (G, G) column covariance
    G = A.shape[1]
    off = (C.sum() - C.diag().sum()) / (G * (G - 1))
    return off.item()


def test_variance_and_cross_covariance_scalings():
    torch.manual_seed(0)
    N, G, M, sigma2 = 2_000_000, 4, 2, 1.0
    s = sigma2 ** 0.5
    r = torch.randn(N, G) * s                 # i.i.d. group rewards (fixed prompt)
    ref = torch.randn(N, M) * s               # independent baseline draws ⟂ r

    A_mws = _advantages_mws(r)
    A_loo = _advantages_loo(r)
    A_ind = _advantages_independent(r, ref)

    tol = 0.01
    # --- conditional advantage variance ---
    assert abs(_var(A_mws) - sigma2 * (G - 1) / G) < tol            # 0.75
    assert abs(_var(A_loo) - sigma2 * G / (G - 1)) < tol            # 1.333
    assert abs(_var(A_ind) - sigma2 * (1 + 1 / M)) < tol            # 1.5

    # --- cross-sample advantage covariance (i≠j) ---
    assert abs(_mean_cross_cov(A_mws) - (-sigma2 / G)) < tol        # -0.25
    assert abs(_mean_cross_cov(A_loo) - (-sigma2 * G / (G - 1) ** 2)) < tol  # -0.444
    assert abs(_mean_cross_cov(A_ind) - (sigma2 / M)) < tol         # +0.5

    # --- variance ratios ---
    assert abs(_var(A_loo) / _var(A_mws) - (G / (G - 1)) ** 2) < 0.02       # 1.78
    assert abs(_var(A_ind) / _var(A_mws) - (1 + 1 / M) * G / (G - 1)) < 0.02  # 2.0


def test_mws_self_inclusion_is_the_only_biased_baseline():
    """MWS baseline correlates with the reward it subtracts (self-inclusion);
    LOO and independent do not — conditional on the prompt. This is the source
    of MWS's gradient bias."""
    torch.manual_seed(1)
    N, G, M, s = 1_000_000, 4, 2, 1.0
    r = torch.randn(N, G) * s
    ref = torch.randn(N, M) * s

    def cond_cov_b_r(b, r):  # within-group Cov(b_i, r_i), averaged over draws
        return ((b - b.mean()) * (r - r.mean())).mean().item()

    b_mws = r.mean(dim=1, keepdim=True).expand_as(r)
    b_loo = (r.sum(dim=1, keepdim=True) - r) / (G - 1)
    b_ind = ref.mean(dim=1, keepdim=True).expand_as(r)

    c_mws = cond_cov_b_r(b_mws, r)
    c_loo = cond_cov_b_r(b_loo, r)
    c_ind = cond_cov_b_r(b_ind, r)

    # MWS shares r_i/G with r_i → Cov ≈ σ²/G = 0.25 > 0; LOO/indep ≈ 0.
    assert abs(c_mws - 1.0 / G) < 0.01
    assert abs(c_loo) < 0.01
    assert abs(c_ind) < 0.01


def test_sequence_level_ess():
    """compute_ess on a row of per-sequence weights: equal → N, one-hot → 1.

    Full-sequence IS yields one weight per rollout, so the meaningful ESS is over
    the batch/group of those weights (passed as a (1, N) row), not the token axis.
    """
    from debiased_grpo.utils import compute_ess
    N = 8
    equal = torch.ones(1, N)
    one_hot = torch.zeros(1, N); one_hot[0, 0] = 1.0
    assert abs(compute_ess(equal).item() - N) < 1e-4
    assert abs(compute_ess(one_hot).item() - 1.0) < 1e-4
