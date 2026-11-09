"""VaR / Expected Shortfall forecasting and backtesting: one-period forecasts from a loss window, GARCH-filtered
tails (McNeil-Frey GARCH-EVT, Barone-Adesi filtered historical simulation) and rolling one-step-ahead backtests
with the Kupiec and Christoffersen tests.

Conventions for the whole module (the same as ``qfkit.risk``):
- LOSSES are positive when money is lost; for returns r pass ``-r``. ``filtered_var_es`` alone takes RETURNS
  (its GARCH model is written for returns); it still reports VaR and ES as positive losses.
- ``alpha`` is the confidence level (0.99), never the tail probability.
- A window of n observations gives the forecast for the NEXT period (n + 1); nothing looks ahead.
- Every convention on which tasks and textbooks differ is a REQUIRED keyword with no default. Pass None for
  one that the chosen method does not use; a ValueError names any that a method needs and did not get.
- Results are in the units of the input (no annualisation). Horizon scaling is done only by
  ``rolling_var_backtest`` (``horizon``, ``sqrt_time``).
- The GARCH(1,1) fit is ``qfkit.garch11_fit`` (arch package conventions); EVT uses ``gpd_fit`` and
  ``pot_var_es``; the tests are ``kupiec_pof``, ``christoffersen_ind`` and ``christoffersen_cc`` (0 ln 0 = 0).

Functions
---------
var_es_forecast       next-period VaR/ES from a loss window: historical, normal, t, Cornish-Fisher, EVT POT,
                      GARCH-filtered (normal, t, EVT, FHS) or RiskMetrics EWMA
filtered_var_es       GARCH(1,1)-filtered VaR/ES: McNeil-Frey GARCH-EVT, FHS, normal or t tail
rolling_var_backtest  trailing-window forecasts for each day, hits, Kupiec and Christoffersen tests

References: McNeil, Frey and Embrechts, Quantitative Risk Management (QRM), 2nd ed. (2015), ch. 2 and 5;
McNeil and Frey (2000), J. Empirical Finance 7, 271-300; Barone-Adesi, Giannopoulos and Vosper (1999),
J. Futures Markets 19, 583-602; Kupiec (1995), J. Derivatives 3, 73-84; Christoffersen (1998), Int. Economic
Review 39, 841-862; RiskMetrics Technical Document (1996); Zangari (1996) for Cornish-Fisher.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .risk import (_QUANTILE_METHODS, _christoffersen_cc, _kupiec_pof, christoffersen_ind, gpd_fit,
                   pot_var_es, var_cornish_fisher, var_es_historical, var_es_normal, var_es_t)
from .riskstats import _arch_backcast, _garch11_filter, ewma_vol, garch11_fit, garch11_forecast

__all__ = [
    "var_es_forecast",
    "filtered_var_es",
    "rolling_var_backtest",
]

_METHODS = ("historical", "normal", "t", "cornish_fisher", "evt_pot", "garch_normal", "garch_t", "garch_evt",
            "garch_fhs", "ewma_normal")
_GARCH_TAIL = {"garch_normal": "normal", "garch_t": "t", "garch_evt": "evt", "garch_fhs": "fhs"}
_GARCH_KW = ("garch_mean", "garch_dist", "input_scale", "vol_at")
# method keywords: (required, optional)
_METHOD_KW = {
    "historical": ((), ("var_floor",)),
    "normal": (("ddof",), ()),
    "t": (("nu", "ddof"), ()),
    "cornish_fisher": (("ddof", "moment_bias"), ()),
    "evt_pot": (("threshold_quantile",), ("gpd_polish",)),
    "garch_normal": (_GARCH_KW, ("params",)),
    "garch_t": (_GARCH_KW, ("params",)),
    "garch_evt": (_GARCH_KW + ("threshold_quantile",), ("params", "gpd_polish")),
    "garch_fhs": (_GARCH_KW, ("params",)),
    "ewma_normal": (("lam", "init"), ()),
}
# which of the three shared conventions each method uses (the others may be None)
_USES = {
    "historical": ("quantile_method", "es_rule"),
    "normal": ("mu_sign",),
    "t": ("mu_sign",),
    "cornish_fisher": ("mu_sign",),
    "evt_pot": ("quantile_method",),
    "garch_normal": (),
    "garch_t": (),
    "garch_evt": ("quantile_method",),
    "garch_fhs": ("quantile_method", "es_rule"),
    "ewma_normal": ("mu_sign",),
}
_ES_RULES = ("ge", "gt", "k_worst")
_TAILS = ("evt", "fhs", "normal", "t")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _vector(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got shape {a.shape})")
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/inf; drop or fill missing rows first")
    return a


def _alphas(alpha):
    """(1-D float array of confidence levels, was_scalar)."""
    a = np.asarray(alpha, dtype=float)
    scalar = a.ndim == 0
    a = np.atleast_1d(a).ravel()
    if a.size == 0 or np.any(~(a > 0.0) | ~(a < 1.0)):
        raise ValueError("alpha is the confidence level and must lie in (0, 1), e.g. 0.99")
    return a, scalar


def _ret(v, scalar):
    v = np.asarray(v, dtype=float)
    return float(v.ravel()[0]) if scalar else v


def _is_int(v):
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _check_qm(qm):
    if qm != "ceil_index" and qm not in _QUANTILE_METHODS:
        raise ValueError(f"quantile_method must be 'ceil_index' or a numpy.quantile method {_QUANTILE_METHODS}")
    return qm


def _check_es_rule(rule):
    if rule not in _ES_RULES:
        raise ValueError(f"es_rule must be one of {_ES_RULES}: 'ge' mean of L >= VaR, 'gt' mean of L > VaR, "
                         "'k_worst' mean of the floor(n(1-alpha)) largest")
    return rule


def _check_mu_sign(mu_sign):
    if isinstance(mu_sign, bool) or not isinstance(mu_sign, numbers.Real) or mu_sign not in (1, 0, -1):
        raise ValueError("mu_sign must be 1 (VaR = mean(L) + z sd: the mean of the LOSSES), 0 (zero mean) or "
                         "-1 (VaR = z sd - mean(L): a 'VaR = -mu + z sigma' formula applied to the loss series)")
    return int(mu_sign)


def _check_fraction(v, name):
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not 0.0 < float(v) < 1.0:
        raise ValueError(f"{name} must be a probability in (0, 1), e.g. 0.9")
    return float(v)


def _check_ddof(ddof, n):
    if not _is_int(ddof) or ddof < 0 or ddof >= n:
        raise ValueError("ddof must be an int, 0 (population) or 1 (sample standard deviation), below the "
                         "window length")
    return int(ddof)


def _quantile(x, q, method):
    """Quantile of a 1-D sample: numpy.quantile(x, q, method=method), or 'ceil_index' = sorted(x)[ceil(q n) - 1]."""
    if method == "ceil_index":
        s = np.sort(x)
        k = max(int(math.ceil(q * s.size)) - 1, 0)
        return float(s[min(k, s.size - 1)])
    return float(np.quantile(x, q, method=method))


def _hist(x, a, qm, rule, var_floor=None):
    """Empirical (VaR, ES) of the sample x at confidence a (reuses risk.var_es_historical)."""
    if qm != "ceil_index" and var_floor is None:
        return var_es_historical(x, a, method=qm, es_rule=rule)
    var = _quantile(x, a, qm)
    if var_floor is not None:
        var = max(var, float(var_floor))
    if rule == "k_worst":       # independent of VaR
        es = var_es_historical(x, a, method="linear", es_rule="k_worst")[1]
    else:
        tail = x[x >= var] if rule == "ge" else x[x > var]
        es = float(tail.mean()) if tail.size else var
    return var, es


def _gpd(y, polish):
    if y.size < 2:
        raise ValueError(f"only {y.size} exceedance(s) above the threshold; lower threshold_quantile or "
                         "lengthen the window")
    if not isinstance(polish, bool):
        raise ValueError("gpd_polish must be True or False")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return gpd_fit(y, polish=polish)


def _method_kw(method, kw, where):
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    req, opt = _METHOD_KW[method]
    unknown = sorted(set(kw) - set(req) - set(opt))
    if unknown:
        raise ValueError(f"{where}: method={method!r} takes the keywords {list(req) + list(opt)}; "
                         f"unknown {unknown}")
    mle_nu = method == "t" and isinstance(kw.get("nu"), str) and kw.get("nu") == "mle"
    missing = [k for k in req if k not in kw and not (mle_nu and k == "ddof")]  # nu='mle' leaves ddof unused
    if missing:
        raise ValueError(f"{where}: method={method!r} needs the keyword(s) {missing} (all of {list(req)})")


def _shared(method, quantile_method, es_rule, mu_sign):
    uses = _USES[method]
    for name, val in (("quantile_method", quantile_method), ("es_rule", es_rule), ("mu_sign", mu_sign)):
        if name in uses and val is None:
            raise ValueError(f"method={method!r} uses {name}; pass it (None only where a method ignores it)")
    return (_check_qm(quantile_method) if "quantile_method" in uses else None,
            _check_es_rule(es_rule) if "es_rule" in uses else None,
            _check_mu_sign(mu_sign) if "mu_sign" in uses else None)


# ----------------------------------------------------------------------------
# GARCH-filtered VaR / ES
# ----------------------------------------------------------------------------
def filtered_var_es(returns, alpha, *, garch_mean, garch_dist, input_scale, tail, threshold_quantile,
                    quantile_method, es_rule, vol_at, params=None, gpd_polish=False):
    """GARCH(1,1)-filtered VaR and Expected Shortfall: McNeil-Frey (2000) GARCH-EVT, Barone-Adesi et al (1999) filtered historical simulation (FHS), or a normal or Student t tail.

    A GARCH(1,1) is fitted to the RETURNS (``qfkit.garch11_fit``, arch conventions) unless ``params`` are given.
    The standardised losses l_t = -(r_t - mu)/sigma_t then give the tail quantile q and tail mean e of the
    innovation, and the loss of the next period is -mu + sigma z, so

        VaR = -mu + sigma * q_alpha(l),      ES = -mu + sigma * ES_alpha(l).

    Parameters
    ----------
    returns : array_like or Series, 1-D, time-ordered, finite (at least 10 when the GARCH is fitted).
        Per-period RETURNS (log or simple, in any unit). VaR/ES come back in the same units, as losses.
    alpha : float or 1-D array in (0, 1). Confidence level(s), e.g. 0.99.
    garch_mean : {'zero', 'constant'}, REQUIRED. r_t = e_t, or r_t = mu + e_t.
    garch_dist : {'normal', 't'}, REQUIRED. Likelihood of the GARCH fit (standardised Student t for 't').
    input_scale : float > 0, REQUIRED (None allowed when ``params`` is given). The fit is run on
        input_scale * returns, e.g. 100 for "fit arch on percent returns", 1 for native units; omega is divided
        by input_scale^2, mu and every volatility by input_scale, so all outputs are in the units of
        ``returns``. (garch11_fit is scale-equivariant, so 1 and 100 agree to optimiser tolerance.)
    tail : {'evt', 'fhs', 'normal', 't'}, REQUIRED. Innovation tail:
        'evt'    McNeil-Frey: POT on l with threshold u = quantile(l, threshold_quantile, quantile_method),
                 exceedances l > u (strict), GPD by MLE on l - u, then the POT VaR/ES (``pot_var_es``);
        'fhs'    Barone-Adesi: empirical quantile of l (``quantile_method``) and tail mean (``es_rule``);
        'normal' q = Phi^-1(alpha), e = phi(q)/(1-alpha);
        't'      standardised (unit-variance) Student t with the fitted nu (needs garch_dist='t' or a 'nu'
                 in params).
    threshold_quantile : float in (0, 1) for tail='evt' (e.g. 0.90); None otherwise. REQUIRED.
    quantile_method : str for tail 'evt' (threshold) and 'fhs' (VaR); None otherwise. REQUIRED.
        Any numpy.quantile method ('linear', 'lower', 'higher', 'inverted_cdf', ...) or 'ceil_index'
        (sorted(l)[ceil(q n) - 1], the smallest l with F_n(l) >= q, as 'inverted_cdf').
    es_rule : {'ge', 'gt', 'k_worst'} for tail 'fhs' (mean of l >= VaR_l, l > VaR_l, or of the k worst);
        None otherwise. REQUIRED.
    vol_at : {'forecast', 'last'}, REQUIRED. sigma used to rescale:
        'forecast' sigma_{T+1} = sqrt(omega + alpha e_T^2 + beta sigma_T^2), the one-step-ahead forecast
                   (a true next-period VaR; what a rolling backtest uses);
        'last'     sigma_T, the last in-sample conditional volatility.
    params : mapping, optional. Fixed GARCH parameters in the units of ``returns``: 'omega', 'alpha', 'beta',
        'mu' (garch_mean='constant'), 'nu' (tail 't'); no fit is run. The variance recursion then starts from
        omega + (alpha + beta) * backcast, backcast as in garch11_fit (0.94-weighted mean of the first 75 squared
        residuals; arch's fix()). With alpha = beta = 0, sigma_t = sqrt(omega) is constant.
    gpd_polish : bool, default False. False fits the GPD with scipy.stats.genpareto.fit(y, floc=0) as is;
        True adds qfkit gpd_fit's L-BFGS-B polish (a slightly lower NLL; VaR differs by about 1e-4 relative).

    Returns
    -------
    dict
        var, es : float (scalar alpha) or ndarray, positive losses in the units of ``returns``.
        sigma : the volatility used (per vol_at); sigma_next : sigma_{T+1}; sigma_last : sigma_T.
        mu : the return mean (0 for 'zero'); params : Series [mu?, omega, alpha, beta, nu?] in native units.
        persistence : alpha + beta; long_run_var : omega/(1 - alpha - beta) (inf if >= 1).
        cond_vol : sigma_t (n,); std_losses : l_t (n,) (Series with the input's index for Series input).
        var_z, es_z : the innovation VaR and ES (q, e above).
        threshold, n_exceed, xi, gpd_beta : the POT fit on l (tail 'evt'; nan / 0 otherwise).

    References
    ----------
    McNeil and Frey (2000) "Estimation of tail-related risk measures for heteroscedastic financial time
    series: an extreme value approach"; Barone-Adesi, Giannopoulos and Vosper (1999) "VaR without
    correlations for portfolios of derivative securities"; McNeil-Frey-Embrechts QRM 2e, Sec. 4.2 and 5.2.

    Example
    -------
    >>> res = filtered_var_es(log_returns, [0.99, 0.995], garch_mean='zero', garch_dist='normal',
    ...                       input_scale=100, tail='evt', threshold_quantile=0.9,
    ...                       quantile_method='lower', es_rule=None, vol_at='last')
    >>> res['var'], res['es'], res['params']['omega'], res['sigma_last']
    """
    index = returns.index if isinstance(returns, pd.Series) else None
    r = _vector(returns, "returns")
    n = r.size
    a, scalar = _alphas(alpha)
    if garch_mean not in ("zero", "constant"):
        raise ValueError("garch_mean must be 'zero' or 'constant'")
    if garch_dist not in ("normal", "t"):
        raise ValueError("garch_dist must be 'normal' or 't' (standardised Student t)")
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}")
    if vol_at not in ("forecast", "last"):
        raise ValueError("vol_at must be 'forecast' (sigma_{T+1}, one step ahead) or 'last' (sigma_T in sample)")
    if tail == "evt":
        tq = _check_fraction(threshold_quantile, "threshold_quantile")
        qm = _check_qm(quantile_method)
    elif tail == "fhs":
        qm = _check_qm(quantile_method)
        rule = _check_es_rule(es_rule)

    nu = math.nan
    if params is None:
        s = input_scale
        if isinstance(s, bool) or not isinstance(s, numbers.Real) or not (math.isfinite(s) and s > 0):
            raise ValueError("input_scale must be a number > 0: 100 to fit on percent returns, 1 for native units")
        s = float(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # standard errors are not used here
            fit = garch11_fit(s * r, mean=garch_mean, dist=garch_dist, cov_type="classic")
        p = fit["params"]
        mu = float(p["mu"]) / s if garch_mean == "constant" else 0.0
        omega, al, be = float(p["omega"]) / (s * s), float(p["alpha"]), float(p["beta"])
        if garch_dist == "t":
            nu = float(p["nu"])
        cond_vol = np.asarray(fit["cond_vol"], dtype=float) / s
        s2_next = float(fit["var_next"]) / (s * s)
    else:
        try:
            omega, al, be = (float(params[k]) for k in ("omega", "alpha", "beta"))
        except (KeyError, TypeError) as exc:
            raise ValueError("params must give 'omega', 'alpha' and 'beta'") from exc
        if not (omega > 0 and al >= 0 and be >= 0):
            raise ValueError("params need omega > 0, alpha >= 0, beta >= 0")
        if garch_mean == "constant":
            if "mu" not in params:
                raise ValueError("garch_mean='constant' with params needs params['mu']")
            mu = float(params["mu"])
        else:
            mu = 0.0
        if "nu" in params:
            nu = float(params["nu"])
        e = r - mu
        bc = _arch_backcast(e)
        s2 = _garch11_filter(e * e, omega, al, be, omega + (al + be) * bc)
        cond_vol = np.sqrt(s2)
        s2_next = float(omega + al * e[-1] ** 2 + be * s2[-1])

    ell = -(r - mu) / cond_vol
    threshold, n_exc, xi, gbeta = math.nan, 0, math.nan, math.nan
    if tail == "normal":
        var_z, es_z = var_es_normal(a, sigma=1.0)
    elif tail == "t":
        if not nu > 2:
            raise ValueError("tail='t' needs a fitted nu > 2: use garch_dist='t' (or give params['nu'])")
        var_z, es_z = var_es_t(a, nu, sd=1.0)
    elif tail == "evt":
        threshold = _quantile(ell, tq, qm)
        y = ell[ell > threshold] - threshold
        g = _gpd(y, gpd_polish)
        xi, gbeta, n_exc = g["xi"], g["beta"], g["n"]
        var_z, es_z = pot_var_es(a, u=threshold, xi=xi, beta=gbeta, n=n, n_u=n_exc)
    else:
        ve = np.array([_hist(ell, ai, qm, rule) for ai in a])
        var_z, es_z = ve[:, 0], ve[:, 1]
    sigma_next = math.sqrt(s2_next)
    sigma = sigma_next if vol_at == "forecast" else float(cond_vol[-1])
    var = -mu + sigma * np.asarray(var_z, dtype=float)
    es = -mu + sigma * np.asarray(es_z, dtype=float)
    names = (["mu"] if garch_mean == "constant" else []) + ["omega", "alpha", "beta"]
    vals = ([mu] if garch_mean == "constant" else []) + [omega, al, be]
    if not math.isnan(nu):
        names.append("nu")
        vals.append(nu)
    pers = al + be
    out = {
        "var": _ret(var, scalar),
        "es": _ret(es, scalar),
        "sigma": sigma,
        "sigma_next": sigma_next,
        "sigma_last": float(cond_vol[-1]),
        "mu": mu,
        "params": pd.Series(vals, index=names, dtype=float),
        "persistence": pers,
        "long_run_var": omega / (1.0 - pers) if pers < 1.0 else math.inf,
        "cond_vol": cond_vol,
        "std_losses": ell,
        "var_z": _ret(var_z, scalar),
        "es_z": _ret(es_z, scalar),
        "threshold": threshold,
        "n_exceed": int(n_exc),
        "xi": xi,
        "gpd_beta": gbeta,
    }
    if index is not None:
        out["cond_vol"] = pd.Series(cond_vol, index=index, name="cond_vol")
        out["std_losses"] = pd.Series(ell, index=index, name="std_losses")
    return out


# ----------------------------------------------------------------------------
# one-period forecast from a window
# ----------------------------------------------------------------------------
def var_es_forecast(losses, alpha, *, method, quantile_method, es_rule, mu_sign, **method_kw):
    """Next-period VaR and Expected Shortfall from a window of losses (McNeil-Frey-Embrechts QRM ch 2 and 5): historical, normal, Student t, Cornish-Fisher, EVT peaks-over-threshold, GARCH-filtered (McNeil-Frey GARCH-EVT, Barone-Adesi FHS) or RiskMetrics EWMA.

    Parameters
    ----------
    losses : array_like or Series, 1-D, time-ordered, finite. LOSSES of the estimation window
        (positive = loss; pass -returns). The forecast is for the period after the last one.
    alpha : float or 1-D array in (0, 1). Confidence level(s), e.g. 0.99.
    method : str, REQUIRED (the keywords each method needs follow it):
        'historical'     VaR = quantile(L, alpha, quantile_method); ES by es_rule.
                         Optional var_floor (e.g. 0.0): VaR = max(VaR, var_floor) before the ES tail is taken.
        'normal'         L ~ N(m, s^2): m = mu_sign * mean(L), s = std(L, ddof); VaR = m + s z,
                         ES = m + s phi(z)/(1-alpha). Needs ddof.
        't'              Student t. nu=float > 2: location m (as 'normal'), standard deviation std(L, ddof)
                         (t scale s sqrt((nu-2)/nu)); nu='mle': (nu, loc, scale) by scipy.stats.t.fit
                         (floc=0 when mu_sign=0; ddof unused). ES = m + scale g(q)/(1-alpha) (nu+q^2)/(nu-1).
        'cornish_fisher' m, s as 'normal'; skew S and excess kurtosis K of the LOSSES by scipy.stats with
                         bias=moment_bias (True: g1, g2; False: the adjusted G1, G2 of pandas/Excel);
                         VaR = m + s z_cf (qfkit.var_cornish_fisher);
                         ES = m + s phi(z)/(1-alpha) [1 + S z/6 + K (z^2-1)/24 - S^2 (2z^2-1)/36], the exact
                         tail average of the Cornish-Fisher quantile (not Boudt et al.'s modified ES).
                         Needs ddof, moment_bias.
        'evt_pot'        POT: u = quantile(L, threshold_quantile, quantile_method), exceedances L > u
                         (strict), GPD MLE of L - u (gpd_polish, default False = scipy genpareto.fit(floc=0)),
                         VaR/ES by qfkit.pot_var_es. Needs threshold_quantile.
        'garch_normal', 'garch_t', 'garch_evt', 'garch_fhs'
                         filtered_var_es on the returns -L with tail 'normal', 't', 'evt' or 'fhs'. Need
                         garch_mean, garch_dist, input_scale, vol_at ('garch_evt' also threshold_quantile);
                         optional params (fixed GARCH parameters), gpd_polish.
        'ewma_normal'    RiskMetrics: s^2 = EWMA variance after the last loss (qfkit.ewma_vol with
                         timing='filter', i.e. lam s^2 + (1-lam) L_T^2), m = mu_sign * mean(L); normal VaR/ES.
                         Needs lam, init ('first_sq' | 'sample_var' | a variance).
    quantile_method : str or None, REQUIRED. Used by 'historical' (VaR), 'evt_pot' and 'garch_evt' (threshold),
        'garch_fhs'; None for the others. A numpy.quantile method ('linear' numpy default, 'lower', 'higher',
        'inverted_cdf', ...) or 'ceil_index' = sorted(L)[ceil(alpha n) - 1] (index clamped at 0), the order
        statistic "smallest x with F_n(x) >= alpha", the same rule as 'inverted_cdf'.
    es_rule : {'ge', 'gt', 'k_worst'} or None, REQUIRED. Used by 'historical' and 'garch_fhs': mean of
        L >= VaR, of L > VaR (VaR itself if none), or of the floor(n(1-alpha)) largest; None for the others.
    mu_sign : {1, 0, -1} or None, REQUIRED. Used by 'normal', 't', 'cornish_fisher', 'ewma_normal':
        location m = mu_sign * mean(L). 1 is the loss mean (VaR = mean(L) + z s, i.e. -mean(r) + z s);
        0 is a zero mean; -1 applies "VaR = -mu + z sigma" to the loss series literally. None for the others.
    **method_kw : the method keywords above (ddof, nu, moment_bias, threshold_quantile, gpd_polish,
        garch_mean, garch_dist, input_scale, vol_at, params, lam, init, var_floor). Unknown or missing ones raise.

    Returns
    -------
    dict with var, es (float for scalar alpha, else ndarray; positive losses) and the estimates:
        historical n; normal/ewma_normal mu, sigma; t mu, scale, nu; cornish_fisher mu, sigma, skew,
        excess_kurt; evt_pot threshold, n_exceed, xi, gpd_beta, n; garch_* every key of filtered_var_es.

    Example
    -------
    >>> L = -log_returns
    >>> var_es_forecast(L, 0.99, method='historical', quantile_method='lower', es_rule='ge', mu_sign=None)
    >>> var_es_forecast(L, 0.99, method='normal', quantile_method=None, es_rule=None, mu_sign=1, ddof=1)
    >>> var_es_forecast(L, [0.99, 0.995], method='evt_pot', quantile_method='lower', es_rule=None,
    ...                 mu_sign=None, threshold_quantile=0.95)
    """
    x = _vector(losses, "losses")
    n = x.size
    a, scalar = _alphas(alpha)
    _method_kw(method, method_kw, "var_es_forecast")
    qm, rule, ms = _shared(method, quantile_method, es_rule, mu_sign)
    kw = method_kw

    if method == "historical":
        floor = kw.get("var_floor")
        if floor is not None and (isinstance(floor, bool) or not isinstance(floor, numbers.Real)):
            raise ValueError("var_floor must be a number (e.g. 0.0) or None")
        ve = np.array([_hist(x, ai, qm, rule, floor) for ai in a])
        return {"var": _ret(ve[:, 0], scalar), "es": _ret(ve[:, 1], scalar), "n": n}

    if method in ("normal", "cornish_fisher") or (method == "t" and kw["nu"] != "mle"):
        ddof = _check_ddof(kw["ddof"], n)
        m = ms * float(x.mean())
        s = float(x.std(ddof=ddof))
    if method == "normal":
        var, es = var_es_normal(a, sigma=s, mu=m)
        return {"var": _ret(var, scalar), "es": _ret(es, scalar), "mu": m, "sigma": s}
    if method == "t":
        nu = kw["nu"]
        if nu == "mle":
            if ms == -1:
                raise ValueError("nu='mle' fits the location: use mu_sign=1 (free) or 0 (floc=0)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df, loc, sc = stats.t.fit(x, floc=0.0) if ms == 0 else stats.t.fit(x)
            df, loc, sc = float(df), float(loc), float(sc)
            var, es = var_es_t(a, df, mu=loc, scale=sc)
            return {"var": _ret(var, scalar), "es": _ret(es, scalar), "mu": loc, "scale": sc, "nu": df}
        if isinstance(nu, bool) or not isinstance(nu, numbers.Real) or not nu > 2:
            raise ValueError("nu must be a number > 2 (the standard deviation is matched) or 'mle'")
        var, es = var_es_t(a, float(nu), mu=m, sd=s)
        return {"var": _ret(var, scalar), "es": _ret(es, scalar), "mu": m,
                "scale": s * math.sqrt(1.0 - 2.0 / float(nu)), "nu": float(nu)}
    if method == "cornish_fisher":
        bias = kw["moment_bias"]
        if not isinstance(bias, bool):
            raise ValueError("moment_bias must be True (scipy default g1, g2) or False (adjusted G1, G2)")
        S = float(stats.skew(x, bias=bias))
        K = float(stats.kurtosis(x, fisher=True, bias=bias))
        var = var_cornish_fisher(a, sigma=s, skew=S, excess_kurt=K, mu=m)
        z = stats.norm.ppf(a)
        es = m + s * stats.norm.pdf(z) / (1.0 - a) * (1.0 + S * z / 6.0 + K * (z * z - 1.0) / 24.0
                                                      - S * S * (2.0 * z * z - 1.0) / 36.0)
        return {"var": _ret(var, scalar), "es": _ret(es, scalar), "mu": m, "sigma": s, "skew": S,
                "excess_kurt": K}
    if method == "evt_pot":
        tq = _check_fraction(kw["threshold_quantile"], "threshold_quantile")
        u = _quantile(x, tq, qm)
        y = x[x > u] - u
        g = _gpd(y, kw.get("gpd_polish", False))
        var, es = pot_var_es(a, u=u, xi=g["xi"], beta=g["beta"], n=n, n_u=g["n"])
        return {"var": _ret(var, scalar), "es": _ret(es, scalar), "threshold": u, "n_exceed": g["n"],
                "xi": g["xi"], "gpd_beta": g["beta"], "n": n}
    if method in _GARCH_TAIL:
        tail = _GARCH_TAIL[method]
        return filtered_var_es(
            -x, a if not scalar else float(a[0]), garch_mean=kw["garch_mean"], garch_dist=kw["garch_dist"],
            input_scale=kw["input_scale"], tail=tail,
            threshold_quantile=kw.get("threshold_quantile") if tail == "evt" else None,
            quantile_method=qm, es_rule=rule, vol_at=kw["vol_at"], params=kw.get("params"),
            gpd_polish=kw.get("gpd_polish", False))
    # ewma_normal
    s = float(np.asarray(ewma_vol(x, lam=kw["lam"], init=kw["init"], timing="filter"))[-1])
    m = ms * float(x.mean())
    var, es = var_es_normal(a, sigma=s, mu=m)
    return {"var": _ret(var, scalar), "es": _ret(es, scalar), "mu": m, "sigma": s}


# ----------------------------------------------------------------------------
# rolling backtest
# ----------------------------------------------------------------------------
_EXACT_H = ("normal", "ewma_normal", "garch_normal")


def _state(method, res, a):
    """What a forecast needs between refits: 'garch' and 'ewma' update their variance with each new loss."""
    if method in _GARCH_TAIL:
        p = res["params"]
        return {"kind": "garch", "mu": res["mu"], "omega": float(p["omega"]), "alpha": float(p["alpha"]),
                "beta": float(p["beta"]), "s2": res["sigma_next"] ** 2, "var_z": res["var_z"], "es_z": res["es_z"]}
    if method == "ewma_normal":
        z = stats.norm.ppf(a)
        return {"kind": "ewma", "mu": res["mu"], "s2": res["sigma"] ** 2, "var_z": float(z),
                "es_z": float(stats.norm.pdf(z) / (1.0 - a))}
    st = {"kind": "static", "var": res["var"], "es": res["es"]}
    if method == "normal":
        z = stats.norm.ppf(a)
        st.update(mu=res["mu"], sigma=res["sigma"], var_z=float(z), es_z=float(stats.norm.pdf(z) / (1.0 - a)))
    return st


def _advance(st, lam, loss_prev):
    if st["kind"] == "garch":
        e = -loss_prev - st["mu"]
        st["s2"] = st["omega"] + st["alpha"] * e * e + st["beta"] * st["s2"]
    elif st["kind"] == "ewma":
        st["s2"] = lam * st["s2"] + (1.0 - lam) * loss_prev * loss_prev


def _values(st, h, sqrt_time):
    """(VaR, ES) for the h-period loss."""
    kind = st["kind"]
    if kind == "static":
        v1, e1 = st["var"], st["es"]
    elif kind == "garch":
        sd = math.sqrt(st["s2"])
        v1, e1 = -st["mu"] + sd * st["var_z"], -st["mu"] + sd * st["es_z"]
    else:
        sd = math.sqrt(st["s2"])
        v1, e1 = st["mu"] + sd * st["var_z"], st["mu"] + sd * st["es_z"]
    if h == 1:
        return v1, e1
    if sqrt_time:
        f = math.sqrt(h)
        return f * v1, f * e1
    if kind == "garch":
        cum = garch11_forecast(st["omega"], st["alpha"], st["beta"], var_next=st["s2"], horizon=h)["cumulative"]
        sd, m = math.sqrt(cum[-1]), -h * st["mu"]
    else:
        sd = (st["sigma"] if kind == "static" else math.sqrt(st["s2"])) * math.sqrt(h)
        m = h * st["mu"]
    return m + sd * st["var_z"], m + sd * st["es_z"]


def rolling_var_backtest(losses, *, window, alpha, method, refit_every, horizon, sqrt_time, quantile_method,
                         es_rule, mu_sign, hit_rule, **method_kw):
    """Rolling-window one-step-ahead VaR and ES backtest with the Kupiec (1995) and Christoffersen (1998) independence and conditional-coverage tests.

    For each t = window, ..., n-1 the forecast for period t uses only losses[t-window : t] (the trailing
    window, excluding t) and is compared with the realised loss of period t; n - window forecasts in all.
    Each forecast is ``var_es_forecast`` on that window, so every method and keyword of var_es_forecast works.

    Parameters
    ----------
    losses : array_like or Series, 1-D, time-ordered, finite. LOSSES (pass -returns).
    window : int >= 2 (>= 10 for GARCH methods), < len(losses), REQUIRED. Window length W.
    alpha : float in (0, 1), REQUIRED. Confidence level; the expected hit rate is 1 - alpha.
    method : str, REQUIRED. A var_es_forecast method; its keywords go in **method_kw (vol_at is fixed to
        'forecast': each forecast is for the next period).
    refit_every : int >= 1, REQUIRED. The model is re-estimated on the trailing window at t = W, W+k, W+2k, ...
        Between refits the estimates are kept: GARCH methods update sigma^2 = omega + alpha e^2 + beta sigma^2
        and 'ewma_normal' updates its variance with each new loss; the other methods keep their forecast.
        1 = refit every period.
    horizon : int >= 1, REQUIRED. h; the realised loss is the sum of losses t..t+h-1 (overlapping for h > 1,
        so hits are then serially dependent by construction).
    sqrt_time : bool, REQUIRED for horizon > 1 (None allowed for 1). True: h-period VaR = sqrt(h) * 1-period
        VaR (and ES); False: exact aggregation, only for 'normal' (h m + sqrt(h) s z), 'ewma_normal' (same)
        and 'garch_normal' (-h mu + z sqrt(sum of the h GARCH variance forecasts)).
    quantile_method, es_rule, mu_sign : REQUIRED, as var_es_forecast (None where the method ignores them).
    hit_rule : {'gt', 'ge'}, REQUIRED. A hit (violation) is loss > VaR ('gt', strict) or loss >= VaR ('ge').
    **method_kw : the method keywords of var_es_forecast.

    Returns
    -------
    dict
        var, es : (n - W,) forecasts for periods W..n-1 (Series on losses.index[W:] for Series input).
        realized : (n - W,) realised (h-period) losses, nan where t + h > n.
        hits : int array (n_obs,) over the forecasts with a realised loss; n_obs = n - W - h + 1.
        n_obs, n_hits, hit_rate, expected_hits = (1 - alpha) n_obs.
        kupiec : dict lr_uc, pvalue, rate (qfkit.kupiec_pof, p = 1 - alpha).
        christoffersen_ind : dict lr_ind, pvalue, pi0, pi1, pi, n00, n01, n10, n11.
        christoffersen_cc : dict lr_uc, lr_ind, lr_cc, p_uc, p_ind, p_cc (LR_cc = LR_uc on all n_obs hits +
            LR_ind, chi2(2)). Every likelihood uses 0 ln 0 = 0.
        var_next, es_next : the forecast for period n from the last window losses[n-W:n].
        n_refits : number of estimations.

    Example
    -------
    >>> bt = rolling_var_backtest(-log_returns, window=250, alpha=0.99, method='garch_evt', refit_every=1,
    ...                           horizon=1, sqrt_time=None, quantile_method='lower', es_rule=None,
    ...                           mu_sign=None, hit_rule='gt', garch_mean='zero', garch_dist='normal',
    ...                           input_scale=100, threshold_quantile=0.9)
    >>> bt['n_hits'], bt['kupiec']['pvalue'], bt['christoffersen_cc']['p_cc']
    """
    index = losses.index if isinstance(losses, pd.Series) else None
    L = _vector(losses, "losses")
    n = L.size
    if not _is_int(window) or window < 2 or window >= n:
        raise ValueError(f"window must be an int in [2, {n - 1}] (len(losses) = {n})")
    W = int(window)
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be ONE confidence level in (0, 1), e.g. 0.99")
    a = float(alpha)
    if not _is_int(refit_every) or refit_every < 1:
        raise ValueError("refit_every must be an int >= 1 (1 = re-estimate every period)")
    if not _is_int(horizon) or horizon < 1:
        raise ValueError("horizon must be an int >= 1")
    h = int(horizon)
    if h > 1:
        if not isinstance(sqrt_time, bool):
            raise ValueError("horizon > 1 needs sqrt_time=True (sqrt(h) scaling) or False (exact aggregation)")
        if not sqrt_time and method not in _EXACT_H:
            raise ValueError(f"sqrt_time=False (exact h-period aggregation) is available for {_EXACT_H}; "
                             "use sqrt_time=True for other methods")
    if hit_rule not in ("gt", "ge"):
        raise ValueError("hit_rule must be 'gt' (loss > VaR) or 'ge' (loss >= VaR)")
    if "vol_at" in method_kw:
        raise ValueError("rolling forecasts always use the one-step-ahead volatility; drop vol_at")
    _method_kw(method, dict(method_kw, **({"vol_at": "forecast"} if method in _GARCH_TAIL else {})),
               "rolling_var_backtest")
    n_obs = n - W - h + 1
    if n_obs < 2:
        raise ValueError(f"need at least 2 forecasts with a realised loss (n - window - horizon + 1 = {n_obs})")
    kw = dict(method_kw)
    if method in _GARCH_TAIL:
        kw["vol_at"] = "forecast"
    lam = float(kw["lam"]) if method == "ewma_normal" else None

    m = n - W + 1                       # forecasts for periods W..n (the last one is var_next)
    var = np.empty(m)
    es = np.empty(m)
    st = None
    n_refits = 0
    for j in range(m):
        t = W + j
        if j % refit_every == 0:
            res = var_es_forecast(L[t - W:t], a, method=method, quantile_method=quantile_method,
                                  es_rule=es_rule, mu_sign=mu_sign, **kw)
            st = _state(method, res, a)
            n_refits += 1
        else:
            _advance(st, lam, L[t - 1])
        var[j], es[j] = _values(st, h, sqrt_time)

    realized = np.full(n - W, np.nan)
    # h = 1: the losses themselves (no cumulative-sum rounding, so a loss equal to VaR stays a tie)
    realized[:n_obs] = L[W:] if h == 1 else np.lib.stride_tricks.sliding_window_view(L[W:], h).sum(axis=1)
    v = var[:n - W]
    hits = (realized[:n_obs] > v[:n_obs]) if hit_rule == "gt" else (realized[:n_obs] >= v[:n_obs])
    hits = hits.astype(np.int64)
    p = 1.0 - a
    x = int(hits.sum())
    kup = {k: float(val) for k, val in _kupiec_pof(x, n_obs, p).items()}  # any alpha in (0, 1)
    ind = christoffersen_ind(hits)
    cc = _christoffersen_cc(hits, p)
    out = {
        "var": v,
        "es": es[:n - W],
        "realized": realized,
        "hits": hits,
        "n_obs": n_obs,
        "n_hits": x,
        "hit_rate": x / n_obs,
        "expected_hits": p * n_obs,
        "kupiec": kup,
        "christoffersen_ind": ind,
        "christoffersen_cc": cc,
        "var_next": float(var[-1]),
        "es_next": float(es[-1]),
        "n_refits": n_refits,
    }
    if index is not None:
        idx = index[W:]
        out["var"] = pd.Series(out["var"], index=idx, name="var")
        out["es"] = pd.Series(out["es"], index=idx, name="es")
        out["realized"] = pd.Series(realized, index=idx, name="realized")
        out["hits"] = pd.Series(hits, index=idx[:n_obs], name="hits")
    return out
