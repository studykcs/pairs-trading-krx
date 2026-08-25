"""A Gaussian Hidden Markov Model (Baum-Welch + scaled forward-backward).

Implemented directly rather than pulled from hmmlearn for two reasons: the
package ships no wheel for this Python and needs a C++ toolchain to build,
and the one property that matters most for a trading backtest - decoding
with *filtered* (causal) probabilities rather than smoothed ones - is easy
to get wrong through a library's default API, which usually hands back
`predict()` = Viterbi over the whole sequence, i.e. every label informed by
the future. See `filter_proba` vs `smooth_proba` below.

Model
-----
K hidden states, multivariate Gaussian emissions with diagonal covariance
(features here are on very different scales - a z-score and a volatility -
and a full covariance over a few hundred observations overfits quickly).

    pi      (K,)      initial state distribution
    A       (K, K)    row-stochastic transition matrix
    means   (K, D)
    vars    (K, D)    diagonal covariance, variance-floored

Why an HMM rather than the Gaussian mixture in gmm_strategy.py: a mixture
treats each day as an independent draw, so a single quiet day inside a
turbulent stretch can flip the label back to "calm" and re-open a position.
An HMM's transition matrix explicitly prices regime *persistence* - staying
put is likelier than switching - so labels are stickier and the strategy
stops whipsawing between regimes on one-day noise. That persistence is the
entire reason to prefer it here, and it is a modeling assumption worth
stating, not a free improvement.

Numerical approach: emissions are computed in log space and shifted by
their per-timestep max before exponentiating (the shift cancels out of the
posteriors and is added back into the log-likelihood), then the standard
scaled forward-backward recursions run in linear space.
"""

from __future__ import annotations

import numpy as np

VAR_FLOOR = 1e-6


def _row_normalize(P: np.ndarray, n_states: int) -> np.ndarray:
    """Rows to sum to 1, falling back to a uniform distribution on rows that
    underflowed to all-zero rather than emitting NaN.

    An observation far in the tail of every state's Gaussian can drive a
    whole alpha row to 0 even with per-timestep scaling. Dividing then gives
    0/0 = NaN, which silently propagates into the caller's regime label; a
    uniform row instead says what is actually true - the model cannot
    distinguish the states here - and leaves the caller's threshold to
    reject it.
    """
    total = P.sum(axis=1, keepdims=True)
    degenerate = (total <= 0) | ~np.isfinite(total)
    safe = np.where(degenerate, 1.0, total)
    out = P / safe
    out[degenerate.ravel()] = 1.0 / n_states
    return out


def _log_gaussian(X: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(T, D), (K, D), (K, D) -> (T, K) log N(x_t | mu_k, diag(var_k))."""
    # (T, 1, D) - (1, K, D) -> (T, K, D)
    diff = X[:, None, :] - means[None, :, :]
    quad = (diff ** 2) / variances[None, :, :]
    logdet = np.log(2 * np.pi * variances).sum(axis=1)  # (K,)
    return -0.5 * (quad.sum(axis=2) + logdet[None, :])


class GaussianHMM:
    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: int = 0,
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.variances: np.ndarray | None = None
        self.loglik_: float = -np.inf

    # -- internals -------------------------------------------------------
    def _emission(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (B, shift): B is a (T, K) non-negative emission matrix
        scaled per timestep, and `shift` the (T,) log constants removed."""
        logB = _log_gaussian(X, self.means, self.variances)
        shift = logB.max(axis=1)
        return np.exp(logB - shift[:, None]), shift

    def _forward(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, K = B.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)

        a = self.pi * B[0]
        scale[0] = a.sum()
        if scale[0] <= 0:
            scale[0] = 1e-300
        alpha[0] = a / scale[0]

        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            scale[t] = a.sum()
            if scale[t] <= 0:
                scale[t] = 1e-300
            alpha[t] = a / scale[t]
        return alpha, scale

    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, K = B.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        T, D = X.shape
        K = self.n_states
        # Seed means from random observations, variances from the global
        # spread - a k-means seed would be marginally better but adds a
        # dependency for very little on windows this short.
        idx = rng.choice(T, size=K, replace=(T < K))
        self.means = X[idx].astype(float).copy()
        self.means += rng.normal(scale=1e-3, size=self.means.shape)
        self.variances = np.tile(X.var(axis=0) + VAR_FLOOR, (K, 1))
        self.A = np.full((K, K), 1.0 / K)
        self.pi = np.full(K, 1.0 / K)

    def _fit_once(self, X: np.ndarray, rng: np.random.Generator) -> float:
        self._init_params(X, rng)
        T, D = X.shape
        K = self.n_states
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            B, shift = self._emission(X)
            alpha, scale = self._forward(B)
            beta = self._backward(B, scale)

            loglik = float(np.log(scale).sum() + shift.sum())

            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-300
            gamma /= gamma_sum

            # xi_t(i,j) propto alpha_t(i) a_ij b_j(x_{t+1}) beta_{t+1}(j) / scale_{t+1}
            xi_sum = np.zeros((K, K))
            for t in range(T - 1):
                num = (
                    alpha[t][:, None]
                    * self.A
                    * (B[t + 1] * beta[t + 1])[None, :]
                    / scale[t + 1]
                )
                xi_sum += num

            self.pi = gamma[0] / gamma[0].sum()
            row = xi_sum.sum(axis=1, keepdims=True)
            row[row == 0] = 1e-300
            self.A = xi_sum / row

            w = gamma.sum(axis=0)
            w[w == 0] = 1e-300
            self.means = (gamma.T @ X) / w[:, None]
            diff = X[:, None, :] - self.means[None, :, :]
            self.variances = np.einsum("tk,tkd->kd", gamma, diff ** 2) / w[:, None]
            self.variances = np.maximum(self.variances, VAR_FLOOR)

            if abs(loglik - prev_ll) < self.tol:
                prev_ll = loglik
                break
            prev_ll = loglik

        return prev_ll

    # -- public API ------------------------------------------------------
    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, dtype=float)
        best = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.random_state + i)
            try:
                ll = self._fit_once(X, rng)
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                continue
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, self.pi.copy(), self.A.copy(),
                        self.means.copy(), self.variances.copy())
        if best is None:
            raise RuntimeError("HMM failed to fit on every restart")
        self.loglik_, self.pi, self.A, self.means, self.variances = best
        return self

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """(T, K) CAUSAL state posteriors: P(state_t | x_1..x_t).

        Forward pass only. This is the one to use for any trading decision -
        it conditions on the past alone, so the label for day t could
        genuinely have been known on day t.
        """
        X = np.asarray(X, dtype=float)
        B, _ = self._emission(X)
        alpha, _ = self._forward(B)
        return _row_normalize(alpha, self.n_states)

    def smooth_proba(self, X: np.ndarray) -> np.ndarray:
        """(T, K) SMOOTHED posteriors: P(state_t | x_1..x_T).

        Uses the whole sequence, future included. Correct for describing
        history (what regime was that period, in hindsight?), and
        look-ahead-biased for backtesting a decision rule. Provided for the
        in-sample regime description in the CLI output, never for signals.
        """
        X = np.asarray(X, dtype=float)
        B, _ = self._emission(X)
        alpha, scale = self._forward(B)
        beta = self._backward(B, scale)
        return _row_normalize(alpha * beta, self.n_states)
