"""Hierarchical Bayesian model for partner-level conversion, written in numpy.

Model (partner j, n_j applications, y_j drawdowns):
    y_j ~ Binomial(n_j, expit(theta_j))
    theta_j ~ Normal(mu, tau^2)            # partial pooling across partners
    mu ~ Normal(0, 3^2)
    tau^2 ~ InverseGamma(2, 0.5)           # weakly informative

Sampler: Metropolis-within-Gibbs. theta_j by random-walk Metropolis (step
size adapted during burn-in), mu and tau^2 by exact Gibbs updates.
Several chains are run so convergence can be checked with R-hat and ESS.
"""
import numpy as np


def _log_expit(x):
    return -np.logaddexp(0.0, -x)


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_chain(y, n, n_iter=6000, burn=2000, seed=0, mu_sd=3.0, tau_prior=(2.0, 0.5)):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    n = np.asarray(n, float)
    J = len(y)
    theta = np.log((y + 0.5) / (n - y + 0.5)) + rng.normal(0, 0.1, J)
    mu, tau2 = theta.mean(), max(theta.var(), 0.05)
    step = np.full(J, 0.3)
    acc = np.zeros(J)
    keep = n_iter - burn
    th_draws = np.empty((keep, J))
    mu_draws = np.empty(keep)
    tau_draws = np.empty(keep)
    a0, b0 = tau_prior

    def logpost(t, mu_, tau2_):
        return y * _log_expit(t) + (n - y) * _log_expit(-t) - 0.5 * (t - mu_) ** 2 / tau2_

    for it in range(n_iter):
        prop = theta + step * rng.normal(size=J)
        accept = np.log(rng.random(J)) < logpost(prop, mu, tau2) - logpost(theta, mu, tau2)
        theta = np.where(accept, prop, theta)
        acc += accept
        if it < burn and (it + 1) % 50 == 0:  # adapt towards ~44% acceptance
            step *= np.exp(acc / 50 - 0.44)
            acc[:] = 0
        prec = J / tau2 + 1 / mu_sd**2
        mu = rng.normal(theta.sum() / tau2 / prec, prec**-0.5)
        tau2 = 1.0 / rng.gamma(a0 + J / 2, 1.0 / (b0 + 0.5 * np.sum((theta - mu) ** 2)))
        if it >= burn:
            k = it - burn
            th_draws[k] = theta
            mu_draws[k] = mu
            tau_draws[k] = np.sqrt(tau2)
    return th_draws, mu_draws, tau_draws


def rhat(chains):
    """Gelman-Rubin R-hat for an (m_chains, n_draws) array."""
    chains = np.asarray(chains)
    m, n = chains.shape
    if m < 2:  # R-hat needs at least two chains
        return float("nan")
    B = n * chains.mean(1).var(ddof=1)
    W = chains.var(1, ddof=1).mean()
    return float(np.sqrt(((n - 1) / n * W + B / n) / W))


def ess(x):
    """Effective sample size of one chain (Geyer initial positive sequence)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    s = 0.0
    for t in range(1, n - 1, 2):
        pair = acf[t] + acf[t + 1]
        if pair < 0:
            break
        s += pair
    return n / (1 + 2 * s)


def fit(y, n, n_chains=4, **kw):
    """Run several chains; return pooled draws and diagnostics."""
    out = [sample_chain(y, n, seed=s, **kw) for s in range(n_chains)]
    th = np.concatenate([o[0] for o in out])
    mu = np.stack([o[1] for o in out])
    tau = np.stack([o[2] for o in out])
    diag = {
        "rhat_mu": rhat(mu),
        "rhat_tau": rhat(tau),
        "ess_mu": float(sum(ess(c) for c in mu)),
        "ess_tau": float(sum(ess(c) for c in tau)),
    }
    return {"p": _expit(th), "mu": mu.ravel(), "tau": tau.ravel(), "diag": diag}


def summarise(p_draws, level: float = 0.90):
    a = (1 - level) / 2
    return (
        p_draws.mean(0),
        np.quantile(p_draws, a, axis=0),
        np.quantile(p_draws, 1 - a, axis=0),
    )
