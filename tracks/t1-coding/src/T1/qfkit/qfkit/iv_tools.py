"""Explicit implied-volatility algorithm variants.

Li ATM cubic: S. Li (2005), A new formula for computing implied volatility,
Applied Mathematics and Computation 170, 611-625, doi:10.1016/j.amc.2004.12.034.
"""
import numpy as np
from .bsm import bsm_price, bsm_greeks

__all__ = ['iv_newton', 'iv_li_atm']


def iv_newton(price, S, K, T, r, *, kind, q, initial, tol, maxiter, criterion, vega_floor):
    """Newton-Raphson inversion of BSM prices with explicit failure diagnostics.

    Parameters
    ----------
    price : finite option price, same currency as S and K.
    S, K : strictly positive spot and strike. Numeric inputs broadcast.
    T : strictly positive maturity in years.
    r : continuous annual rate in decimals.
    kind : 'call' or 'put'.
    q : continuous annual dividend yield in decimals.
    initial : strictly positive initial annual volatility, scalar or broadcastable.
    tol : positive absolute tolerance, in volatility or price units per criterion.
    maxiter : positive integer number of Newton updates.
    criterion : 'step' for abs(new_sigma-old_sigma)<tol, 'price' for abs(residual)<tol.
    vega_floor : nonnegative absolute vega below which inversion stops.

    Returns
    -------
    dict : volatility, converged, iterations, price_residual and status arrays
        (zero-dimensional for scalar input). Status is converged, maxiter,
        out_of_bounds, low_vega or invalid_step. Failed initial prices have NaN
        volatility; other failures retain the last positive iterate. There is
        no damping, clipping or switch to Brent. Inspect convergence before use.

    Example
    -------
    >>> p=bsm_price(100,103,1,.02,.24,kind='call',q=.01)
    >>> iv_newton(p,100,103,1,.02,kind='call',q=.01,initial=.3,tol=1e-10,maxiter=50,criterion='step',vega_floor=1e-12)['converged'].item()
    True
    """
    arr=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (price,S,K,T,r,q,initial)])
    target,spot,strike,time,rate,div,start=arr
    if any(np.any(~np.isfinite(x)) for x in arr) or any(np.any(x<=0) for x in (spot,strike,time,start)):
        raise ValueError('finite inputs and positive S/K/T/initial required')
    if kind not in ('call','put') or criterion not in ('step','price'):
        raise ValueError('invalid kind or convergence criterion')
    if not np.isfinite(tol) or tol<=0 or not np.isfinite(vega_floor) or vega_floor<0:
        raise ValueError('positive tolerance and nonnegative vega floor required')
    if not isinstance(maxiter,(int,np.integer)) or maxiter<1: raise ValueError('maxiter must be positive integer')
    sd=spot*np.exp(-div*time);kd=strike*np.exp(-rate*time)
    low=np.maximum(sd-kd,0) if kind=='call' else np.maximum(kd-sd,0)
    high=sd if kind=='call' else kd
    valid=np.asarray((target>low)&(target<high))
    vol=np.where(valid,start,np.nan);converged=np.zeros(target.shape,dtype=bool)
    iterations=np.zeros(target.shape,dtype=int);status=np.full(target.shape,'out_of_bounds',dtype='<U16')
    status[valid]='maxiter'
    active=valid.copy()
    for _ in range(maxiter):
        if not np.any(active): break
        idx=np.flatnonzero(active)
        args=[x.ravel()[idx] for x in (spot,strike,time,rate,div)]
        s,k,t,rt,d=args;v=vol.ravel()[idx]
        residual=np.asarray(bsm_price(s,k,t,rt,v,kind=kind,q=d))-target.ravel()[idx]
        if criterion=='price':
            done=np.abs(residual)<tol
            converged.ravel()[idx[done]]=True;status.ravel()[idx[done]]='converged';active.ravel()[idx[done]]=False
            idx=idx[~done]
            if len(idx)==0: continue
            s,k,t,rt,d=[x.ravel()[idx] for x in (spot,strike,time,rate,div)]
            v=vol.ravel()[idx];residual=np.asarray(bsm_price(s,k,t,rt,v,kind=kind,q=d))-target.ravel()[idx]
        vega=np.asarray(bsm_greeks(s,k,t,rt,v,kind=kind,q=d)['vega'])
        bad=(~np.isfinite(vega))|(vega<=vega_floor)
        active.ravel()[idx[bad]]=False;status.ravel()[idx[bad]]='low_vega'
        good=~bad;idx=idx[good]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            new=v[good]-residual[good]/vega[good]
        iterations.ravel()[idx]+=1
        invalid=(~np.isfinite(new))|(new<=0)
        active.ravel()[idx[invalid]]=False;status.ravel()[idx[invalid]]='invalid_step'
        good_step=~invalid;idx=idx[good_step];old=vol.ravel()[idx].copy();new=new[good_step]
        vol.ravel()[idx]=new
        if criterion=='step':
            done=np.abs(new-old)<tol
            converged.ravel()[idx[done]]=True;status.ravel()[idx[done]]='converged';active.ravel()[idx[done]]=False
    residual=np.full(target.shape,np.nan)
    idx=np.flatnonzero(valid)
    if len(idx):
        s,k,t,rt,d,v=[x.ravel()[idx] for x in (spot,strike,time,rate,div,vol)]
        residual.ravel()[idx]=np.asarray(bsm_price(s,k,t,rt,v,kind=kind,q=d))-target.ravel()[idx]
    if criterion=='price':
        done=active&(np.abs(residual)<tol)
        converged[done]=True;status[done]='converged'
    return {'volatility':vol,'converged':converged,'iterations':iterations,'price_residual':residual,'status':status}


def iv_li_atm(price, S, T, *, q):
    """Li cubic/trigonometric implied-volatility approximation at forward ATM.

    Parameters
    ----------
    price : nonnegative finite call/put price (equal at forward ATM).
    S : strictly positive spot, broadcastable with other inputs.
    T : strictly positive maturity in years.
    q : continuous annual dividend yield in decimals.

    Returns
    -------
    float or ndarray : annual decimal volatility from the small nonnegative
        root of 2*x-x**3/3=sqrt(2*pi)*price/(S*exp(-q*T)), x=sigma*sqrt(T)/2.
        NaN when the cubic's arccos argument is outside [-1,1]. No clamping or
        non-ATM extension. Caller must check K*exp(-r*T)=S*exp(-q*T); spot ATM
        and forward ATM differ when carry is nonzero.

    Example
    -------
    >>> round(iv_li_atm(7.9655674554,100,1,q=0.),4)
    0.2
    """
    price,S,T,q=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (price,S,T,q)])
    if any(np.any(~np.isfinite(x)) for x in (price,S,T,q)) or np.any(price<0) or np.any(S<=0) or np.any(T<=0):
        raise ValueError('finite inputs, nonnegative price and positive S/T required')
    z=3*np.sqrt(2*np.pi)*price/(4*np.sqrt(2)*S*np.exp(-q*T))
    with np.errstate(invalid='ignore'):
        x=2*np.sqrt(2)*np.cos((np.arccos(-z)+4*np.pi)/3)
    vol=np.where(price==0,0.,2*x/np.sqrt(T))
    return float(vol) if vol.ndim==0 else vol
