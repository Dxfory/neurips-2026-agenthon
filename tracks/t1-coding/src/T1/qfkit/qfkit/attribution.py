"""Arithmetic attribution and covariance-based risk scaling with explicit inputs."""
import numpy as np

__all__ = ['brinson_fachler', 'vol_target_weights']


def vol_target_weights(weights, covariance, *, target_vol, periods_per_year, max_scale):
    """Covariance portfolio volatility targeting by a capped weight multiplier.

    Parameters
    ----------
    weights : finite 1-D base weights; their gross exposure is not normalized.
    covariance : finite symmetric positive-semidefinite per-period return covariance.
    target_vol : nonnegative annual volatility target, a decimal.
    periods_per_year : positive number of return periods in one year.
    max_scale : nonnegative finite maximum multiplier on the base weights.
        This is not a gross-exposure cap unless the base weights have unit gross.

    Returns
    -------
    dict : weights, scale, base_vol, achieved_vol and capped. Covariance includes
        off-diagonal terms. Zero base risk raises for a positive target; a zero
        target gives zero weights. No covariance estimation or timing is inferred.

    Example
    -------
    >>> vol_target_weights([.6,-.4],[[.0004,.0001],[.0001,.0009]],target_vol=.12,periods_per_year=252,max_scale=2.)
    """
    w=np.asarray(weights,dtype=float);c=np.asarray(covariance,dtype=float)
    target,periods,cap=map(float,(target_vol,periods_per_year,max_scale))
    if w.ndim!=1 or len(w)==0 or c.shape!=(len(w),len(w)) or np.any(~np.isfinite(w)) or np.any(~np.isfinite(c)):
        raise ValueError('finite weights and matching covariance required')
    if not np.all(np.isfinite([target,periods,cap])) or min(target,cap)<0 or periods<=0:
        raise ValueError('nonnegative target/cap and positive periods_per_year required')
    if not np.allclose(c,c.T,rtol=1e-10,atol=1e-14): raise ValueError('covariance must be symmetric')
    if np.linalg.eigvalsh(c).min() < -1e-12*max(1e-12,float(np.max(np.abs(c)))):
        raise ValueError('covariance must be positive semidefinite')
    base=float(np.sqrt(max(0.,float(w@c@w))*periods))
    if base==0 and target>0: raise ValueError('positive target cannot be reached from zero base risk')
    requested=target/base if base>0 else 0.;scale=min(requested,cap)
    return {'weights':w*scale,'scale':float(scale),'base_vol':base,
            'achieved_vol':base*scale,'capped':bool(requested>cap)}


def brinson_fachler(portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns):
    """Brinson-Fachler allocation, selection and interaction by sector and period.

    Parameters
    ----------
    portfolio_weights, benchmark_weights : beginning-of-period weights. Last axis
        is sectors; each set must sum to one. Negative weights are permitted.
    portfolio_returns, benchmark_returns : sector simple decimal returns, with
        the same sector axis; inputs broadcast. Include cash explicitly: give
        cash the same return on both sides and zero benchmark weight if absent.

    Returns
    -------
    dict : allocation, selection and interaction arrays, plus portfolio_return,
        benchmark_return and active_return with the sector axis summed out.
        Effects sum to the single-period active return. Summing effects over
        periods is arithmetic attribution, not a compounded-return linking rule.
        Rebalancing, weight drift and multiperiod linking are caller decisions.

    Example
    -------
    >>> brinson_fachler([.7,.3], [1.,0.], [.12,.003], [.1,.003])['active_return']
    """
    wp,wb,rp,rb=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in
        (portfolio_weights,benchmark_weights,portfolio_returns,benchmark_returns)])
    if wp.ndim==0 or wp.shape[-1]==0 or any(np.any(~np.isfinite(x)) for x in (wp,wb,rp,rb)):
        raise ValueError('inputs must be finite with a nonempty sector axis')
    if any(not np.allclose(w.sum(-1),1.,rtol=0.,atol=1e-10) for w in (wp,wb)):
        raise ValueError('beginning-of-period weights must sum to one')
    port=(wp*rp).sum(-1);bench=(wb*rb).sum(-1)
    return {'allocation':(wp-wb)*(rb-bench[...,None]),'selection':wb*(rp-rb),
            'interaction':(wp-wb)*(rp-rb),'portfolio_return':port,
            'benchmark_return':bench,'active_return':port-bench}
