"""Reflection-principle probabilities, convex call slices, and local-volatility paths.

Brownian inputs describe X(t)=mu*t+sigma*W(t), X(0)=0; for GBM use
log-barrier distance and log-drift. Surface inputs are currency call prices.
No random generator is created: simulation consumes the caller's normal draws.

References: Karatzas and Shreve, Brownian Motion and Stochastic Calculus,
reflection principle; Dupire (1994), Pricing with a Smile; Glasserman (2004),
Monte Carlo Methods in Financial Engineering. Numerical derivatives deliberately
use numpy.gradient twice, including on irregular grids.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special

__all__ = ['brownian_hit_cdf', 'brownian_hit_pdf', 'brownian_joint_cdf',
           'dupire_gradient_surface', 'project_call_slice', 'localvol_paths',
           'barrier_digital_price', 'gap_option_price', 'dupire_fd_surface']


def _out(value):
    a = np.asarray(value)
    return float(a) if a.ndim == 0 else a


def _inputs(barrier, t, mu, sigma):
    a = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (barrier, t, mu, sigma)])
    if any(np.any(~np.isfinite(v)) for v in a):
        raise ValueError('inputs must be finite')
    if np.any(a[1] < 0) or np.any(a[3] <= 0):
        raise ValueError('time must be nonnegative and sigma strictly positive')
    return a


def _direction(mu, direction):
    if direction not in ('up', 'down'):
        raise ValueError("direction must be 'up' or 'down'")
    return mu if direction == 'up' else -mu


def brownian_hit_cdf(barrier, t, mu, sigma, *, direction):
    """Brownian first-passage cumulative probability at a barrier distance from zero.

    Parameters
    ----------
    barrier : nonnegative distance; log(H/S0) up or log(S0/L) down for GBM.
    t : nonnegative time in years, broadcast with all numeric arguments.
    mu, sigma : annual log-drift and positive diffusion, in the original direction.
    direction : 'up' or 'down'; downward passage internally negates mu.

    Returns
    -------
    float or ndarray : P(tau <= t), including mass 1 at time zero if barrier=0.
        Away-pointing drift gives a defective distribution. E[tau]=barrier/mu
        only for positive toward-barrier drift; unconditional mean is infinite
        for nonpositive toward-barrier drift and a positive barrier.

    Example
    -------
    >>> brownian_hit_cdf(0.1, 1., 0.02, 0.2, direction='up')
    """
    b, t, mu, sigma = _inputs(barrier, t, mu, sigma)
    if np.any(b < 0):
        raise ValueError('barrier is a nonnegative distance')
    drift = _direction(mu, direction)
    safe_t = np.where(t > 0, t, 1.)
    sd = sigma*np.sqrt(safe_t)
    # The reflected term is <=1; log-CDF avoids exp(large)*Phi(very negative).
    reflected = np.exp(np.minimum(0., 2*drift*b/sigma**2 + special.log_ndtr((-b-drift*safe_t)/sd)))
    value = special.ndtr((drift*safe_t-b)/sd) + reflected
    return _out(np.where(b == 0, 1., np.where(t == 0, 0., np.clip(value, 0., 1.))))


def brownian_hit_pdf(barrier, t, mu, sigma, *, direction):
    """Continuous first-passage density for drifted Brownian motion.

    Parameters
    ----------
    barrier, t, mu, sigma : as brownian_hit_cdf; time in years, sigma positive.
    direction : 'up' or 'down', using the original process drift mu.

    Returns
    -------
    float or ndarray : density per year. Zero at t=0 and for barrier=0; the
        latter has a point mass at zero that is not represented by a density.

    Example
    -------
    >>> brownian_hit_pdf(0.1, 0.5, 0.02, 0.2, direction='up')
    """
    b, t, mu, sigma = _inputs(barrier, t, mu, sigma)
    if np.any(b < 0):
        raise ValueError('barrier is a nonnegative distance')
    drift = _direction(mu, direction)
    safe_t = np.where(t > 0, t, 1.)
    safe_b = np.where(b > 0, b, 1.)
    logp = np.log(safe_b)-np.log(sigma)-.5*np.log(2*np.pi)-1.5*np.log(safe_t)
    logp -= (b-drift*safe_t)**2/(2*sigma**2*safe_t)
    return _out(np.where((t > 0) & (b > 0), np.exp(logp), 0.))


def brownian_joint_cdf(x, maximum, t, mu, sigma):
    """Joint CDF P(X(t)<=x, max[0,t] X<=maximum) for drifted Brownian motion.

    Parameters
    ----------
    x, maximum : terminal and running-maximum levels, in log-space for GBM.
    t, mu, sigma : nonnegative time, annual log-drift and positive diffusion.
        All numeric inputs broadcast; X starts at zero.

    Returns
    -------
    float or ndarray : joint probability; zero for maximum<0. For x>maximum,
        the terminal restriction is redundant. At t=0 X and its maximum are zero.

    Example
    -------
    >>> brownian_joint_cdf(0., 0.1, 1., 0.02, 0.2)
    """
    b, t, mu, sigma = _inputs(maximum, t, mu, sigma)
    x, b, t, mu, sigma = np.broadcast_arrays(np.asarray(x,dtype=float), b,t,mu,sigma)
    if np.any(~np.isfinite(x)):
        raise ValueError('x must be finite')
    safe_t = np.where(t > 0,t,1.)
    a = np.minimum(x, b)
    sd = sigma*np.sqrt(safe_t)
    reflected = np.exp(np.minimum(0., 2*mu*b/sigma**2 + special.log_ndtr((a-2*b-mu*safe_t)/sd)))
    value = np.clip(special.ndtr((a-mu*safe_t)/sd)-reflected,0.,1.)
    return _out(np.where(t == 0, ((x >= 0)&(b >= 0)).astype(float), np.where(b <= 0,0.,value)))


def _grid(values, name, minimum):
    a = np.asarray(values,dtype=float)
    if a.ndim != 1 or a.size < minimum or np.any(~np.isfinite(a)) or np.any(a <= 0) or np.any(np.diff(a)<=0):
        raise ValueError(f'{name} must contain at least {minimum} positive increasing finite points')
    return a


def barrier_digital_price(S, K, H, T, r, sigma, *, q, kind, payoff, direction, knock, cash=1.):
    """Continuously monitored single-barrier digital price by reflection.

    Parameters
    ----------
    S, K, H : positive spot, terminal trigger and barrier; numeric inputs broadcast.
    T : nonnegative maturity in years; a barrier touched at inception is hit.
    r, sigma : annual continuous rate and strictly positive volatility.
    q : annual continuous dividend yield.
    kind : 'call' (S_T>K) or 'put' (S_T<K), with strict terminal trigger.
    payoff : 'cash' (cash units at maturity) or 'asset' (S_T at maturity).
    direction : 'up' or 'down'. Continuous monitoring includes inception.
    knock : 'in' or 'out'. No rebate; knock-in uses vanilla minus knock-out.
    cash : nonnegative cash payoff, ignored for asset digitals; default 1.

    Returns
    -------
    float or ndarray : present value. Asset digitals use the stock-numeraire
        drift shift sigma**2; cash digitals use the risk-neutral log drift.
        Not a discrete-monitoring or touch-time payment formula.

    Example
    -------
    >>> barrier_digital_price(100., 105., 80., 1., .03, .2, q=.01, kind='call', payoff='cash', direction='down', knock='out')
    """
    if kind not in ('call','put') or payoff not in ('cash','asset'):
        raise ValueError('kind must be call/put and payoff cash/asset')
    if direction not in ('up','down') or knock not in ('in','out'):
        raise ValueError('direction must be up/down and knock in/out')
    s,k,h,t,rate,vol,div,c=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (S,K,H,T,r,sigma,q,cash)])
    if any(np.any(~np.isfinite(x)) for x in (s,k,h,t,rate,vol,div,c)):
        raise ValueError('inputs must be finite')
    if np.any(s<=0)|np.any(k<=0)|np.any(h<=0)|np.any(t<0)|np.any(vol<=0)|np.any(c<0):
        raise ValueError('prices and volatility positive; maturity and cash nonnegative')
    mu=rate-div-.5*vol**2+(vol**2 if payoff=='asset' else 0.)
    scale=s*np.exp(-div*t) if payoff=='asset' else c*np.exp(-rate*t)
    x=np.log(k/s);b=np.log(h/s)
    safe_t=np.where(t>0,t,1.)
    put=special.ndtr((x-mu*safe_t)/(vol*np.sqrt(safe_t)))
    vanilla=np.where(t>0,1-put if kind=='call' else put,(s>k if kind=='call' else s<k).astype(float))
    if direction=='up':
        survive=brownian_joint_cdf(b,b,t,mu,vol)
        below=brownian_joint_cdf(x,b,t,mu,vol)
        out=survive-below if kind=='call' else below
        hit=s>=h
    else:
        survive=brownian_joint_cdf(-b,-b,t,-mu,vol)
        above=brownian_joint_cdf(-x,-b,t,-mu,vol)
        out=above if kind=='call' else survive-above
        hit=s<=h
    out=np.where(hit,0.,np.where(t==0,vanilla,np.clip(out,0.,vanilla)))
    return _out(scale*(out if knock=='out' else vanilla-out))


def gap_option_price(S, payoff_strike, trigger, T, r, sigma, *, q, kind):
    """Gap option with distinct payoff strike and exercise trigger.

    Parameters
    ----------
    S, payoff_strike, trigger : positive spot, payoff strike and terminal trigger.
    T, r, sigma : nonnegative years, continuous annual rate, annual volatility.
    q : continuous annual dividend yield.
    kind : 'call' pays (S_T-payoff_strike)*1[S_T>trigger]; 'put' pays
        (payoff_strike-S_T)*1[S_T<trigger]. The payment may be negative.

    Returns
    -------
    float or ndarray : discounted asset digital minus strike times cash digital
        for a call, opposite sign for a put. Inputs broadcast as digital_price.
        At zero variance and forward=trigger, digital_price uses the half-mass
        limiting convention rather than strict exercise at a point mass.

    Example
    -------
    >>> gap_option_price(100., 105., 98., .5, .03, .2, q=.01, kind='call')
    """
    from .exotics import digital_price
    strike=np.asarray(payoff_strike,dtype=float)
    if np.any(~np.isfinite(strike)) or np.any(strike<=0):
        raise ValueError('payoff_strike must be finite and positive')
    asset=digital_price(S,trigger,T,r,sigma,q=q,kind=kind,payoff='asset')
    cash=digital_price(S,trigger,T,r,sigma,q=q,kind=kind,payoff='cash')
    return _out((1 if kind=='call' else -1)*(asset-strike*cash))


def dupire_gradient_surface(strikes, maturities, calls, *, rate, dividend, variance_floor, vol_cap):
    """Dupire local volatility with numpy.gradient applied twice on the strike grid.

    Parameters
    ----------
    strikes, maturities : positive increasing grids, each of length >=3.
    calls : finite call prices, shape (n_maturities,n_strikes), in currency.
    rate, dividend : constant continuously compounded annual decimal rates.
    variance_floor, vol_cap : positive lower variance and upper volatility,
        with variance_floor<=vol_cap**2. For an undiscounted forward call
        surface use rate=dividend=0 and a consistent fixed-forward strike grid.

    Returns
    -------
    dict : dC_dT, dC_dK, d2C_dK2, raw_variance, local_vol and regularized masks.
        Formula is 2*(C_T+(r-q)*K*C_K+q*C)/(K**2*C_KK). Nonpositive curvature
        or nonfinite variance is replaced by variance_floor; other variances
        are clipped between variance_floor and vol_cap**2.

    Example
    -------
    >>> result = dupire_gradient_surface(K, T, C, rate=0., dividend=0., variance_floor=1e-6, vol_cap=2.)
    """
    k=_grid(strikes,'strikes',3);t=_grid(maturities,'maturities',3)
    c=np.asarray(calls,dtype=float)
    if c.shape != (len(t),len(k)) or np.any(~np.isfinite(c)):
        raise ValueError('calls must be a finite maturity-by-strike matrix')
    r,q,floor,cap=map(float,(rate,dividend,variance_floor,vol_cap))
    if not np.all(np.isfinite([r,q,floor,cap])) or floor<=0 or cap<=0 or floor>cap**2:
        raise ValueError('rates and bounds must be finite, 0<variance_floor<=vol_cap**2')
    ct=np.gradient(c,t,axis=0,edge_order=2)
    ck=np.gradient(c,k,axis=1,edge_order=2)
    ckk=np.gradient(ck,k,axis=1,edge_order=2)
    with np.errstate(divide='ignore',invalid='ignore'):
        raw=2*(ct+(r-q)*k[None,:]*ck+q*c)/(k[None,:]**2*ckk)
    valid=(ckk>0)&np.isfinite(raw)
    bounded=np.clip(np.where(valid,raw,floor),floor,cap**2)
    return {'dC_dT':ct,'dC_dK':ck,'d2C_dK2':ckk,'raw_variance':raw,
            'local_vol':np.sqrt(bounded),'regularized':(~valid)|(raw<floor)|(raw>cap**2)}


def dupire_fd_surface(strikes, maturities, calls, *, rate, dividend, variance_floor,
                      min_denominator, time_edge_order):
    """Dupire local volatility with an interior three-point strike second difference.

    Parameters
    ----------
    strikes, maturities : strictly positive increasing grids, at least three points.
    calls : finite maturity-by-strike matrix of discounted call prices.
    rate, dividend : constant continuous annual rate and dividend yield.
    variance_floor : nonnegative floor after dividing the Dupire numerator.
    min_denominator : nonnegative cutoff on K**2*C_KK; at or below it returns NaN.
    time_edge_order : 1 or 2 for numpy.gradient time-boundary differences.
        Interior time/first-strike derivatives use centered three-point formulas.

    Returns
    -------
    dict : local_vol, raw_variance, dC_dT, dC_dK, d2C_dK2 arrays. The first and
        last strike columns have NaN second derivatives and local volatilities;
        no one-sided curvature is invented. Unequal strike spacings are supported.
        No upper volatility cap or smoothing is applied.

    Example
    -------
    >>> out = dupire_fd_surface(K,T,C,rate=.03,dividend=0.,variance_floor=1e-10,min_denominator=1e-15,time_edge_order=1)
    """
    k=_grid(strikes,'strikes',3);t=_grid(maturities,'maturities',3);c=np.asarray(calls,dtype=float)
    if c.shape!=(len(t),len(k)) or np.any(~np.isfinite(c)):
        raise ValueError('calls must be a finite maturity-by-strike matrix')
    r,q,floor,cut=map(float,(rate,dividend,variance_floor,min_denominator))
    if not np.all(np.isfinite([r,q,floor,cut])) or min(floor,cut)<0 or time_edge_order not in (1,2):
        raise ValueError('finite rates, nonnegative bounds and time_edge_order 1 or 2 required')
    ct=np.gradient(c,t,axis=0,edge_order=time_edge_order)
    ck=np.gradient(c,k,axis=1,edge_order=1)
    ckk=np.full(c.shape,np.nan)
    left=np.diff(k)[:-1];right=np.diff(k)[1:]
    ckk[:,1:-1]=2*((c[:,2:]-c[:,1:-1])/right-(c[:,1:-1]-c[:,:-2])/left)/(left+right)
    den=k[None,:]**2*ckk;num=2*(ct+(r-q)*k[None,:]*ck+q*c)
    raw=np.divide(num,den,out=np.full(c.shape,np.nan),where=den>cut)
    return {'local_vol':np.sqrt(np.maximum(raw,floor)),'raw_variance':raw,
            'dC_dT':ct,'dC_dK':ck,'d2C_dK2':ckk}


def project_call_slice(strikes, calls, *, forward, discount):
    """Least-squares projection of a call-price slice onto strike no-arbitrage bounds.

    Parameters
    ----------
    strikes : at least two positive increasing strikes.
    calls : finite prices at those strikes, in currency.
    forward : positive forward price for this expiry, in the strike currency.
    discount : positive discount factor to this expiry (dimensionless).

    Returns
    -------
    ndarray : nearest equal-weight squared-error prices satisfying discounted
        intrinsic/forward bounds, nonpositive slopes bounded below by -discount,
        and increasing slopes (convexity). Does not enforce calendar constraints.

    Example
    -------
    >>> project_call_slice([80,100,120], [25,30,2], forward=102., discount=.98)
    """
    k=_grid(strikes,'strikes',2);c=np.asarray(calls,dtype=float)
    f,d=map(float,(forward,discount))
    if c.shape != k.shape or np.any(~np.isfinite(c)) or not np.isfinite(f*d) or f<=0 or d<=0:
        raise ValueError('calls must match strikes; forward and discount must be finite positive')
    x=k/f;y=c/(d*f);lo=np.maximum(1-x,0.)
    slopes=np.zeros((len(k)-1,len(k)))
    for i,dx in enumerate(np.diff(x)):
        slopes[i,i]=-1/dx;slopes[i,i+1]=1/dx
    rows=np.vstack([slopes,np.diff(slopes,axis=0)])
    lower=np.r_[-np.ones(len(k)-1),np.zeros(len(k)-2)]
    upper=np.r_[np.zeros(len(k)-1),np.full(len(k)-2,np.inf)]
    def feasible(v,tol):
        av=rows@v
        return np.all(v>=lo-tol) and np.all(v<=1+tol) and np.all(av>=lower-tol) and np.all(av<=upper+tol)
    if feasible(y,1e-12):
        return c.copy()
    result=optimize.minimize(lambda v:.5*np.sum((v-y)**2),lo.copy(),jac=lambda v:v-y,
                             method='SLSQP',bounds=optimize.Bounds(lo,np.ones(len(k))),
                             constraints=[optimize.LinearConstraint(rows,lower,upper)],
                             options={'ftol':1e-12,'maxiter':2000})
    if not result.success or not feasible(result.x,1e-8):
        raise RuntimeError('convex call-slice projection did not converge')
    return result.x*(d*f)


def localvol_paths(S0, times, *, rate, dividend, local_vol, normals):
    """Log-Euler paths with state-dependent local volatility and caller-provided normals.

    Parameters
    ----------
    S0 : finite positive initial price.
    times : positive increasing observation times in years (time zero is implicit).
    rate, dividend : finite constant continuous annual decimal rates.
    local_vol : callable(t,spots) returning finite nonnegative annual volatilities,
        evaluated at the PREVIOUS step's time and spot. It owns interpolation
        and extrapolation choices; scalar output is broadcast to all paths.
    normals : finite array (n_paths,n_steps). Supply antithetic or legacy draws
        explicitly when required. No random draws occur inside this function.

    Returns
    -------
    ndarray : shape (n_paths,n_steps+1), including S0. Barrier monitoring is a
        separate operation: select the prescribed observation columns, not all
        substeps, when barriers are monitored less often than the simulation.

    Example
    -------
    >>> paths = localvol_paths(100., [.5,1.], rate=.03, dividend=0., local_vol=lambda t,s:.2, normals=Z)
    """
    times=_grid(times,'times',1);z=np.asarray(normals,dtype=float)
    s,r,q=map(float,(S0,rate,dividend))
    if not np.all(np.isfinite([s,r,q])) or s<=0 or z.ndim!=2 or z.shape[1]!=len(times) or z.shape[0]<1 or np.any(~np.isfinite(z)):
        raise ValueError('invalid spot, rates or normals shape/values')
    out=np.empty((z.shape[0],len(times)+1));out[:,0]=s
    for i,(previous,dt) in enumerate(zip(np.r_[0.,times[:-1]],np.diff(np.r_[0.,times]))):
        vol=np.broadcast_to(np.asarray(local_vol(float(previous),out[:,i]),dtype=float),out[:,i].shape)
        if np.any(~np.isfinite(vol)) or np.any(vol<0):
            raise ValueError('local_vol must return finite nonnegative values')
        out[:,i+1]=out[:,i]*np.exp((r-q-.5*vol**2)*dt+vol*np.sqrt(dt)*z[:,i])
        if np.any(~np.isfinite(out[:,i+1])) or np.any(out[:,i+1]<=0):
            raise FloatingPointError('path overflow/underflow; inspect parameters and time step')
    return out
