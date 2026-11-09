"""Explicit adapters for the installed arch package; no substitute estimator.

Reference: https://arch.readthedocs.io/en/latest/univariate/introduction.html
The optional dependency is imported only when fitting, never on qfkit import.
"""
import numpy as np
from scipy import stats

__all__=['arch_garch_fit','rolling_garch_var']


def arch_garch_fit(returns, *, mean, dist, input_scale, rescale, cov_type,
                   vol='GARCH', p=1, o=0, q=1, power=2., options=None, starting_values=None):
    """Fit with arch.arch_model and expose both raw fit parameters and original-unit risks.

    arch on decimal returns with input_scale=1 and rescale=False stops at its starting values
    and still reports convergence_flag 0. garch11_fit accepts any units and reports native
    units (same arch likelihood, fitted on internally rescaled data). With arch_garch_fit, use
    input_scale=100 and read params_native.

    Parameters
    ----------
    returns : finite nonconstant 1-D returns, in the caller's desired output units.
    mean : arch mean model 'Zero' or 'Constant' (case-insensitive).
    dist : arch innovation distribution 'normal', 't', 'skewt' or 'ged' (case-insensitive;
        aliases 'gaussian', 'studentst', 'skewstudent').
    input_scale : positive multiplier applied before fitting, e.g. 100 for decimals to percent.
    rescale : True, False or None, passed directly to arch; None warns without rescaling.
    cov_type : arch covariance estimator 'robust' or 'classic'.
    vol, p, o, q, power : arch volatility specification, defaults GARCH(1,0,1), power=2.
    options : optional optimizer options dict, passed to arch's SLSQP fit.
    starting_values : optional parameters IN FIT UNITS, passed unchanged to arch.

    Returns
    -------
    dict : params (arch's Series IN FIT UNITS), conditional_volatility, residuals,
        standardized_residuals, next_mean, next_variance, effective_scale,
        loglikelihood, aic, bic, converged, convergence_flag and arch_result.
        Risks and residuals are in original return units, variances squared.
        effective_scale=input_scale*arch_result.scale; raw omega is NOT converted.
        No estimator or optimizer fallback; inspect converged before using a fit.
        Also params_native (params in original return units: mu / s, omega / s**power,
        s = effective_scale; other labels unchanged), persistence (alpha + beta
        + 0.5 gamma summed over lags; GARCH power 2 only, else NaN), long_run_variance
        (native omega / (1 - persistence); inf when persistence >= 1) and at_start_values
        (True when the fit stayed at arch's starting values, or at starting_values when
        given). converged = convergence_flag == 0 and not at_start_values.

    Example
    -------
    >>> r=arch_garch_fit(x,mean='Zero',dist='normal',input_scale=100.,rescale=False,cov_type='robust')
    >>> r['next_variance']  # one-period forecast, in original return units squared
    """
    x=np.asarray(returns,dtype=float);scale=float(input_scale)
    if x.ndim!=1 or len(x)<2 or np.any(~np.isfinite(x)) or np.ptp(x)==0:
        raise ValueError('returns must be a nonconstant finite vector with at least two values')
    if not np.isfinite(scale) or scale<=0 or (rescale is not None and not isinstance(rescale,(bool,np.bool_))):
        raise ValueError('positive finite input_scale and boolean/None rescale required')
    mean_name=_MEANS.get(str(mean).lower());dist_name=_DISTS.get(str(dist).lower())
    if mean_name is None or dist_name is None:
        raise ValueError('unsupported mean or distribution; use arch directly for other model families')
    if cov_type not in ('robust','classic'): raise ValueError('cov_type must be robust or classic')
    from arch import arch_model
    fit=arch_model(x*scale,mean=mean_name,vol=vol,p=p,o=o,q=q,power=power,dist=dist_name,rescale=rescale).fit(
        disp='off',update_freq=0,cov_type=cov_type,options=options,starting_values=starting_values,
        show_warning=False)
    effective=scale*float(fit.scale)
    forecast=fit.forecast(horizon=1,method='analytic',reindex=False)
    sigma=np.asarray(fit.conditional_volatility)/effective
    residual=np.asarray(fit.resid)/effective
    at_start=_at_start_values(fit,starting_values)
    native=fit.params.copy().astype(float)
    if 'mu' in native.index: native['mu']=native['mu']/effective
    if 'omega' in native.index: native['omega']=native['omega']/effective**float(power)
    persistence=np.nan;long_run=np.nan
    if str(vol).lower()=='garch' and float(power)==2. and 'omega' in native.index:
        persistence=float(sum(native[k] for k in native.index if k.startswith(('alpha[','beta[')))
                          +0.5*sum(native[k] for k in native.index if k.startswith('gamma[')))
        long_run=float(native['omega']/(1.-persistence)) if persistence<1 else np.inf
    return {'params':fit.params.copy(),'conditional_volatility':sigma,'residuals':residual,
            'standardized_residuals':np.asarray(fit.std_resid),
            'next_mean':float(forecast.mean.iloc[-1,0])/effective,
            'next_variance':float(forecast.variance.iloc[-1,0])/(effective**2),
            'effective_scale':effective,'loglikelihood':float(fit.loglikelihood),
            'aic':float(fit.aic),'bic':float(fit.bic),'converged':bool(fit.convergence_flag==0 and not at_start),
            'convergence_flag':int(fit.convergence_flag),'arch_result':fit,'params_native':native,
            'persistence':persistence,'long_run_variance':long_run,'at_start_values':at_start}


_MEANS={'zero':'Zero','constant':'Constant'}
_DISTS={'normal':'normal','gaussian':'normal','t':'t','studentst':'t','skewt':'skewt','skewstudent':'skewt',
        'ged':'ged'}


def _at_start_values(fit,starting_values):
    """True when arch's fitted parameters are its own start vector (mean and volatility parts), or the caller's
    starting_values, to rtol 1e-8: SLSQP stopped without a step and still reports convergence_flag 0."""
    try:
        if starting_values is not None:
            sv=np.asarray(starting_values,dtype=float).ravel()
        else:
            model=fit.model  # the data arch fitted (after its own rescale)
            start=np.asarray(model.starting_values(),dtype=float).ravel()
            sv=np.r_[start,np.asarray(model.volatility.starting_values(model.resids(start)),dtype=float).ravel()]
        got=np.asarray(fit.params.values,dtype=float)
    except Exception:  # an arch model without these hooks: no stall evidence, never a crash
        return False
    if sv.size==0 or sv.size>got.size: return False
    return bool(np.allclose(got[:sv.size],sv,rtol=1e-8,atol=0))


def rolling_garch_var(returns, *, window, origins, alpha, horizon, horizon_method,
                      mean, dist, input_scale, rescale, cov_type, on_failure, options=None):
    """Past-window arch GARCH(1,1) VaR forecasts with explicit square-root-time scaling.

    Parameters
    ----------
    returns : finite 1-D chronological arithmetic/log returns; no conversion is made.
    window : integer >=2, number of returns used at each fit.
    origins : integer vector; origin i fits returns[i-window:i], forecasts i onward.
        window<=i<=len(returns). Selected origins avoid unnecessary repeated fits.
    alpha : confidence level in (0,1), not tail probability.
    horizon : positive integer number of return periods.
    horizon_method : 'sqrt_time' only: one-step variance times horizon and mean times horizon.
        This is an explicit approximation, NOT a sum of GARCH variance forecasts.
    mean : 'Zero' or 'Constant', passed to arch_garch_fit.
    dist : 'normal' or 't'; t innovations use arch's standardized-t variance convention.
    input_scale : positive input multiplier, passed to arch_garch_fit.
    rescale : bool/None, passed to arch_garch_fit.
    cov_type : 'robust' or 'classic', passed to arch_garch_fit.
    on_failure : 'raise' or 'nan'; failed/nonconverged fits never become valid forecasts.
        A fit stalled at arch's starting values (convergence_flag 0, at_start_values) is
        NaN under 'nan'; under 'raise' it keeps its forecast, marked converged False with
        a 'stalled' status, and does not raise.
    options : optional optimizer options, passed to arch_garch_fit.

    Returns
    -------
    dict : origins, var (positive-loss convention), realized_loss, observed,
        exceedance, converged and status arrays. Realized loss is -sum of the
        horizon returns, NaN when incomplete; exceedance is false when forecast
        or realization is absent, so use observed AND converged when counting.
        For simple returns the realized horizon value is additive, not compounded.

    Example
    -------
    >>> r=rolling_garch_var(x,window=100,origins=[100,120],alpha=.99,horizon=1,horizon_method='sqrt_time',mean='Zero',dist='normal',input_scale=100.,rescale=False,cov_type='robust',on_failure='raise')
    """
    x=np.asarray(returns,dtype=float);idx=np.asarray(origins)
    if x.ndim!=1 or np.any(~np.isfinite(x)) or not isinstance(window,(int,np.integer)) or window<2:
        raise ValueError('finite return vector and integer window>=2 required')
    if idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer) or np.any(idx<window) or np.any(idx>len(x)):
        raise ValueError('integer origins must have complete estimation windows')
    if not isinstance(horizon,(int,np.integer)) or horizon<1 or not 0<alpha<1:
        raise ValueError('positive integer horizon and confidence alpha in (0,1) required')
    dist=_DISTS.get(str(dist).lower(),dist)
    if horizon_method!='sqrt_time' or dist not in ('normal','t') or on_failure not in ('raise','nan'):
        raise ValueError('supported methods: sqrt_time, normal/t, on_failure raise/nan')
    values=np.full(len(idx),np.nan);realized=np.full(len(idx),np.nan)
    converged=np.zeros(len(idx),dtype=bool);status=[]
    for row,i in enumerate(idx):
        if i+horizon<=len(x): realized[row]=-float(x[i:i+horizon].sum())
        try:
            fit=arch_garch_fit(x[i-window:i],mean=mean,dist=dist,input_scale=input_scale,
                               rescale=rescale,cov_type=cov_type,options=options)
            stalled=bool(fit.get('at_start_values')) and fit.get('convergence_flag',0)==0
            if not fit['converged'] and not stalled: raise RuntimeError('arch fit did not converge')
            if stalled and on_failure=='nan': raise RuntimeError('arch fit stalled at its starting values')
            if dist=='normal': z=stats.norm.ppf(alpha)
            else:
                nu=float(fit['params']['nu'])
                if nu<=2: raise ValueError('standardized Student t requires nu>2')
                z=stats.t.ppf(alpha,nu)*np.sqrt((nu-2)/nu)
            values[row]=-fit['next_mean']*horizon+z*np.sqrt(fit['next_variance']*horizon)
            if not np.isfinite(values[row]): raise FloatingPointError('nonfinite VaR forecast')
            converged[row]=not stalled
            status.append('stalled: arch fit stayed at its starting values' if stalled else 'ok')
        except ImportError:
            raise
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            if on_failure=='raise': raise
            values[row]=np.nan;status.append(type(exc).__name__+': '+str(exc))
    observed=np.isfinite(realized)
    return {'origins':idx.copy(),'var':values,'realized_loss':realized,'observed':observed,
            'exceedance':observed&converged&(realized>values),'converged':converged,'status':status}
