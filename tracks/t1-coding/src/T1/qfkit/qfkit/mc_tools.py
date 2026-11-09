"""Per-path GBM Greek estimators with shared, externally supplied normal draws.

References: Broadie and Glasserman (1996), Estimating Security Price Derivatives
Using Simulation; Columbia IEOR E4703, Estimating the Greeks:
https://www.columbia.edu/~mh2078/MonteCarlo/MCS_Greeks_MasterSlides.pdf
The library reports derivatives in unit volatility/rate and years; output
conventions such as one percentage point or one day are a caller's conversion.
"""
from __future__ import annotations

import numpy as np

__all__ = ['gbm_greek_samples', 'gbm_fd_greek_samples', 'mc_estimate', 'correlated_gbm_terminal']


def correlated_gbm_terminal(spot, T, mu, sigma, *, correlation, normals, method):
    """Correlated multi-asset geometric Brownian terminal values from supplied draws.

    Parameters
    ----------
    spot : positive 1-D initial prices, one per asset.
    T : nonnegative horizon in years.
    mu, sigma : annual arithmetic drifts and nonnegative volatilities, scalar
        or broadcast to spot. Risk-neutral drift is r-q, chosen by the caller.
    correlation : symmetric positive-semidefinite asset correlation matrix.
    normals : independent standard-normal draws, shape (paths,assets).
    method : 'cholesky' requires positive-definite correlation; 'eigh' uses
        symmetric square root and also allows singular correlations. Different
        roots give different seeded paths; choose the prescribed method.

    Returns
    -------
    ndarray : terminal prices, shape (paths,assets). The library draws no random
        numbers. Returns spot broadcast over paths at T=0.

    Example
    -------
    >>> s = correlated_gbm_terminal([100,90],1,.03,[.2,.3],correlation=[[1,.4],[.4,1]],normals=Z,method='cholesky')
    """
    s=np.asarray(spot,dtype=float);z=np.asarray(normals,dtype=float);c=np.asarray(correlation,dtype=float)
    t=float(T)
    if s.ndim!=1 or len(s)==0 or np.any(~np.isfinite(s)) or np.any(s<=0) or not 0<=t<np.inf:
        raise ValueError('positive finite spot vector and nonnegative horizon required')
    m=np.broadcast_to(np.asarray(mu,dtype=float),s.shape);v=np.broadcast_to(np.asarray(sigma,dtype=float),s.shape)
    if np.any(~np.isfinite(m)) or np.any(~np.isfinite(v)) or np.any(v<0):
        raise ValueError('finite drifts and nonnegative volatilities required')
    if c.shape!=(len(s),len(s)) or np.any(~np.isfinite(c)) or not np.allclose(c,c.T,atol=1e-12,rtol=0):
        raise ValueError('correlation must be a finite symmetric asset matrix')
    if not np.allclose(np.diag(c),1.,atol=1e-12,rtol=0) or np.any(np.abs(c)>1+1e-12):
        raise ValueError('invalid correlation entries')
    if z.ndim!=2 or z.shape[1]!=len(s) or z.shape[0]<1 or np.any(~np.isfinite(z)):
        raise ValueError('normals must have shape (paths,assets) and be finite')
    if method=='cholesky':
        root=np.linalg.cholesky(c)
    elif method=='eigh':
        vals,vec=np.linalg.eigh(c)
        if vals.min()<-1e-12: raise ValueError('correlation must be positive semidefinite')
        root=(vec*np.sqrt(np.maximum(vals,0)))@vec.T
    else: raise ValueError('method must be cholesky or eigh')
    out=s*np.exp((m-.5*v*v)*t+np.sqrt(t)*v*(z@root.T))
    if np.any(~np.isfinite(out)) or np.any(out<=0):
        raise FloatingPointError('terminal prices overflowed or underflowed')
    return out


def _setup(S, K, T, r, sigma, q, kind, averaging, normals, theta):
    values = np.asarray([S,K,T,r,sigma,q],dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values[[0,1,2,4]]<=0):
        raise ValueError('spot, strike, maturity and volatility must be finite positive; rates finite')
    if kind not in ('call','put') or averaging not in ('terminal','arithmetic'):
        raise ValueError("kind is call/put and averaging is terminal/arithmetic")
    if theta not in ('calendar','maturity'):
        raise ValueError("theta must be 'calendar' (-dV/dT) or 'maturity' (dV/dT)")
    z=np.asarray(normals,dtype=float)
    if z.ndim==1:
        z=z[:,None]
    if z.ndim!=2 or min(z.shape)<1 or np.any(~np.isfinite(z)):
        raise ValueError('normals must be finite with shape (n_paths,n_steps)')
    return tuple(values),z,1. if kind=='call' else -1.,-1. if theta=='calendar' else 1.


def _path(S,T,r,sigma,q,z):
    t=T*np.arange(1,z.shape[1]+1)/z.shape[1]
    w=np.sqrt(T/z.shape[1])*np.cumsum(z,axis=1)
    path=S*np.exp((r-q-.5*sigma*sigma)*t+sigma*w)
    if np.any(~np.isfinite(path)):
        raise FloatingPointError('GBM path overflow')
    return t,w,path


def _payoff(S,K,T,r,sigma,q,sign,averaging,z):
    _,_,path=_path(S,T,r,sigma,q,z)
    terminal=path[:,-1] if averaging=='terminal' else path.mean(axis=1)
    return np.exp(-r*T)*np.maximum(sign*(terminal-K),0.)


def gbm_greek_samples(S, K, T, r, sigma, *, q, kind, averaging, normals, theta):
    """Pathwise and likelihood-ratio Greeks under geometric Brownian motion.

    Parameters
    ----------
    S, K : positive scalar spot and strike in the same currency.
    T : positive scalar maturity in years.
    r : scalar continuously compounded annual interest rate.
    sigma : positive scalar annual volatility as a decimal.
    q : scalar continuously compounded annual dividend yield as a decimal.
    kind : 'call' or 'put'.
    averaging : 'terminal' European or 'arithmetic' equally spaced Asian,
        excluding the initial spot from the average.
    normals : standard-normal draws (paths,steps), generated by the caller.
        A 1-D array means one step; European terminal Z aggregates all steps.
    theta : 'calendar' gives -dV/dT per year; 'maturity' gives dV/dT.

    Returns
    -------
    dict : price samples, and pathwise/lr dictionaries with delta/gamma/vega/
        theta/rho sample arrays. Vega and rho are per unit decimal change.
        Pathwise gamma and LR theta are None; Asian pathwise reports delta
        and vega only. Asian LR spot derivatives use the FIRST transition's
        score, while rate/volatility scores sum over all transitions. Use
        mc_estimate on each sample array for a mean and standard error.

    Example
    -------
    >>> values = gbm_greek_samples(100,100,1,.03,.2,q=.01,kind='call',averaging='terminal',normals=Z,theta='calendar')
    """
    values,z,sign,theta_sign=_setup(S,K,T,r,sigma,q,kind,averaging,normals,theta)
    S,K,T,r,sigma,q=values
    t,w,path=_path(S,T,r,sigma,q,z)
    disc=np.exp(-r*T)
    underlying=path[:,-1] if averaging=='terminal' else path.mean(axis=1)
    payoff=np.maximum(sign*(underlying-K),0.)
    active=sign*(sign*(underlying-K)>0)
    price=disc*payoff
    vega_path=path*(w-sigma*t)
    pw={'delta':disc*active*underlying/S, 'gamma':None,
        'vega':disc*active*(vega_path[:,-1] if averaging=='terminal' else vega_path.mean(axis=1)),
        'theta':None,'rho':None}
    if averaging=='terminal':
        pw['rho']=disc*T*(active*underlying-payoff)
        dterminal=underlying*((r-q-.5*sigma*sigma)+sigma*w[:,-1]/(2*T))
        pw['theta']=theta_sign*disc*(-r*payoff+active*dterminal)
        score_z=w[:,-1]/np.sqrt(T);dt=T
        vol_score=(score_z**2-1)/sigma-score_z*np.sqrt(T)
        rate_score=score_z*np.sqrt(T)/sigma-T
    else:
        score_z=z[:,0];dt=T/z.shape[1]
        vol_score=np.sum((z*z-1)/sigma-z*np.sqrt(dt),axis=1)
        rate_score=np.sum(z*np.sqrt(dt)/sigma,axis=1)-T
    scale=S*sigma*np.sqrt(dt)
    lr={'delta':price*score_z/scale,
        'gamma':price*(score_z**2-1-score_z*sigma*np.sqrt(dt))/(scale*scale),
        'vega':price*vol_score,'theta':None,'rho':price*rate_score}
    return {'price':price,'pathwise':pw,'lr':lr}


def gbm_fd_greek_samples(S, K, T, r, sigma, *, q, kind, averaging, normals,
                         spot_bump, vol_bump, time_bump, rate_bump, theta):
    """Central finite-difference Greeks under geometric Brownian motion.

    Parameters
    ----------
    S, K, T, r, sigma : positive spot, strike, years; annual rate and volatility.
    q : scalar continuously compounded annual dividend yield as a decimal.
    kind : 'call' or 'put'.
    averaging : 'terminal' European or 'arithmetic' Asian excluding initial spot.
    normals : standard-normal draws, shape (paths,steps), reused for every bump.
    theta : 'calendar' gives -dV/dT; 'maturity' gives dV/dT per year.
    spot_bump, vol_bump, time_bump, rate_bump : positive absolute bumps; spot,
        volatility and maturity minus their bumps must remain strictly positive.
        All scenarios reuse the same normals; shortened maturity rescales time.

    Returns
    -------
    dict : delta, gamma, vega, theta and rho arrays, in unit decimal vol/rate
        and per-year theta. Standard errors must be taken over these paired
        differences, not combined from independent scenario standard errors.

    Example
    -------
    >>> g = gbm_fd_greek_samples(100,100,1,.03,.2,q=0.,kind='call',averaging='terminal',normals=Z,spot_bump=.5,vol_bump=.01,time_bump=1/252,rate_bump=.001,theta='calendar')
    """
    values,z,sign,theta_sign=_setup(S,K,T,r,sigma,q,kind,averaging,normals,theta)
    S,K,T,r,sigma,q=values
    hs,hv,ht,hr=map(float,(spot_bump,vol_bump,time_bump,rate_bump))
    if not np.all(np.isfinite([hs,hv,ht,hr])) or min(hs,hv,ht,hr)<=0 or hs>=S or hv>=sigma or ht>=T:
        raise ValueError('bumps must be positive finite, smaller than spot/volatility/maturity')
    def value(s=S,v=sigma,t=T,rate=r):
        return _payoff(s,K,t,rate,v,q,sign,averaging,z)
    plus=value(s=S+hs);minus=value(s=S-hs);base=value()
    return {'delta':(plus-minus)/(2*hs),'gamma':(plus-2*base+minus)/(hs*hs),
            'vega':(value(v=sigma+hv)-value(v=sigma-hv))/(2*hv),
            'theta':theta_sign*(value(t=T+ht)-value(t=T-ht))/(2*ht),
            'rho':(value(rate=r+hr)-value(rate=r-hr))/(2*hr)}


def mc_estimate(samples, *, pair_antithetic):
    """Sample mean, variance and standard error with optional antithetic pairing.

    Parameters
    ----------
    samples : finite one-dimensional discounted payoff or estimator samples.
    pair_antithetic : bool. True expects [original half, antithetic half] and
        averages each pair BEFORE computing the sample variance and error.

    Returns
    -------
    dict : value (mean), variance (ddof=1), se, n_independent and n_samples.
        At least two independent observations (or antithetic pairs) are required.

    Example
    -------
    >>> mc_estimate([1.,2.,3.,4.], pair_antithetic=False)['value']
    2.5
    """
    a=np.asarray(samples,dtype=float)
    if not isinstance(pair_antithetic,(bool,np.bool_)) or a.ndim!=1 or np.any(~np.isfinite(a)):
        raise ValueError('samples must be a finite vector; pair_antithetic must be boolean')
    count=len(a)
    if pair_antithetic:
        if count%2:
            raise ValueError('antithetic samples must have even length')
        a=(a[:count//2]+a[count//2:])/2
    if len(a)<2:
        raise ValueError('at least two independent observations are needed')
    variance=float(np.var(a,ddof=1))
    return {'value':float(np.mean(a)),'variance':variance,'se':float(np.sqrt(variance/len(a))),
            'n_independent':len(a),'n_samples':count}
