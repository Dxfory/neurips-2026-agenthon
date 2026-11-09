"""Independent CIR-factor characteristic functions and static variance replication.

Two-factor model: Chiarella and Ziveyi, UTS QFR Research Paper 292 (2011),
https://www.uts.edu.au/globalassets/sites/default/files/qfr-archive-03/QFR-rp292.pdf
The European component factors across independent Heston variance drivers.
Uses the numerically stable one-factor characteristic function in volsurf.
"""
import functools
import numpy as np
from scipy import integrate
from .volsurf import _heston_cf

__all__ = ['multifactor_heston_price', 'variance_replication', 'synthetic_forward_bounds', 'finite_difference_risk']


def finite_difference_risk(pricer, parameters, *, steps, mixed):
    """Central parameter bumps for a scalar pricer's gradient and Hessian.

    Parameters
    ----------
    pricer : callable accepting one parameter vector and returning a finite scalar.
        It must be deterministic at each point. For Monte Carlo, close over the
        same random draws at all bumps. The caller owns parameter domain checks.
    parameters : finite nonempty 1-D parameter vector.
    steps : strictly positive finite bumps with the same shape, in parameter units.
    mixed : bool; True evaluates cross derivatives with the four-corner stencil.

    Returns
    -------
    dict : value, gradient and hessian. Off-diagonal Hessian entries are NaN
        when mixed=False. Derivatives are per parameter unit; e.g. negate the
        maturity derivative for calendar theta and divide annual theta by the
        requested day-count denominator. No Greek units are inferred.

    Example
    -------
    >>> finite_difference_risk(lambda x:x[0]**2,[2.],steps=[.01],mixed=False)['gradient']
    array([4.])
    """
    x=np.asarray(parameters,dtype=float);h=np.asarray(steps,dtype=float)
    if x.ndim!=1 or len(x)==0 or h.shape!=x.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(h)) or np.any(h<=0):
        raise ValueError('finite parameter vector and matching positive step vector required')
    if not isinstance(mixed,(bool,np.bool_)): raise ValueError('mixed must be boolean')
    def value(point):
        v=np.asarray(pricer(point.copy()),dtype=float)
        if v.ndim!=0 or not np.isfinite(v): raise ValueError('pricer must return a finite scalar')
        return float(v)
    v0=value(x);gradient=np.empty(len(x));hessian=np.full((len(x),len(x)),np.nan)
    basis=np.diag(h)
    for i in range(len(x)):
        plus=value(x+basis[i]);minus=value(x-basis[i])
        gradient[i]=(plus-minus)/(2*h[i]);hessian[i,i]=(plus-2*v0+minus)/(h[i]**2)
        if mixed:
            for j in range(i):
                cross=(value(x+basis[i]+basis[j])-value(x+basis[i]-basis[j])
                       -value(x-basis[i]+basis[j])+value(x-basis[i]-basis[j]))/(4*h[i]*h[j])
                hessian[i,j]=hessian[j,i]=cross
    return {'value':v0,'gradient':gradient,'hessian':hessian}


@functools.lru_cache(maxsize=16)
def _legendre(n):
    return np.polynomial.legendre.leggauss(n)


def synthetic_forward_bounds(strike, call_bid, call_ask, put_bid, put_ask, *, discount):
    """Executable synthetic-forward bid and ask from European put-call parity.

    Parameters
    ----------
    strike : positive strike, shared by the paired European call and put.
    call_bid, call_ask, put_bid, put_ask : finite nonnegative option prices,
        same currency and timestamp, with ask>=bid. Numeric inputs broadcast.
    discount : positive discount factor to the common option expiry.

    Returns
    -------
    dict : lower=K+(call_bid-put_ask)/discount and
        upper=K+(call_ask-put_bid)/discount, plus their midpoint. These are
        synthetic forward quotes; no extra borrow, fee, carry or stale-quote
        assumptions are inserted. Cleaning and pairing precede this function.

    Example
    -------
    >>> synthetic_forward_bounds(100,6,6.5,3,3.6,discount=.96)['lower']
    102.5
    """
    k,cb,ca,pb,pa,df=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in
        (strike,call_bid,call_ask,put_bid,put_ask,discount)])
    if any(np.any(~np.isfinite(x)) for x in (k,cb,ca,pb,pa,df)):
        raise ValueError('quotes, strike and discount must be finite')
    if np.any(k<=0) or np.any(df<=0) or any(np.any(x<0) for x in (cb,ca,pb,pa)) or np.any(ca<cb) or np.any(pa<pb):
        raise ValueError('positive strike/discount and nonnegative uncrossed quotes required')
    lower=k+(cb-pa)/df;upper=k+(ca-pb)/df
    return {'lower':lower,'upper':upper,'midpoint':(lower+upper)/2}


def multifactor_heston_price(S, K, T, r, factors, *, q, kind, quadrature, bounds,
                            points=256, limit=200, epsabs=1e-9, epsrel=1e-9):
    """Multi-factor Heston European option price from the P1/P2 probability integrals.

    Parameters
    ----------
    S, K, T, r : spot, strike, nonnegative years and continuous annual rate;
        these inputs and q broadcast. Spot and strike must be positive.
    factors : finite (n_factors,5) array, columns v0,kappa,theta,sigma_v,rho.
        Risk-neutral CIR factors are independent across rows; each row's rho
        couples its own spot shock to its variance shock. v0/theta/sigma_v
        nonnegative, kappa positive, |rho|<=1. No extra market-risk-premium shift.
    q : continuous annual dividend yield.
    kind : 'call' or 'put'; puts directly use (1-P1) and (1-P2).
    quadrature : 'quad' for scipy.integrate.quad, or 'gauss_legendre' using
        np.polynomial.legendre.leggauss; interval is always caller supplied.
    bounds : nonnegative finite lower and strictly greater upper frequency bound.
    points : positive integer Gauss-Legendre order; default 256.
    limit : positive integer maximum quad subinterval count; default 200.
    epsabs, epsrel : positive quad tolerances; default 1e-9.

    Returns
    -------
    float or ndarray : price, without clipping numerical integration errors.
        Pj=1/2+integral(Re(exp(i*u*log(F/K))*phi_j(u)/(i*u)))/pi;
        phi_2=product(phi_factor(u)), phi_1=product(phi_factor(u-i)).
        Caller checks truncation/quadrature convergence for its chosen bounds.

    Example
    -------
    >>> multifactor_heston_price(100,105,1,.03,[[.04,1.2,.04,.3,-.5]],q=.01,kind='call',quadrature='quad',bounds=(1e-8,150))
    """
    p=np.asarray(factors,dtype=float)
    if p.ndim!=2 or p.shape[1]!=5 or p.shape[0]<1 or np.any(~np.isfinite(p)):
        raise ValueError('factors must be a finite n-by-5 array')
    if np.any(p[:,:4]<0) or np.any(p[:,1]<=0) or np.any(np.abs(p[:,4])>1):
        raise ValueError('invalid CIR factor parameters')
    if kind not in ('call','put') or quadrature not in ('quad','gauss_legendre'):
        raise ValueError('invalid option kind or quadrature')
    b=np.asarray(bounds,dtype=float)
    if b.shape!=(2,) or np.any(~np.isfinite(b)) or not 0<=b[0]<b[1]:
        raise ValueError('bounds must be finite and 0<=lower<upper')
    if int(points)!=points or points<1 or int(limit)!=limit or limit<1:
        raise ValueError('points and limit must be positive integers')
    if not np.all(np.isfinite([epsabs,epsrel])) or min(epsabs,epsrel)<=0:
        raise ValueError('quadrature tolerances must be finite and positive')
    arrays=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (S,K,T,r,q)])
    if any(np.any(~np.isfinite(x)) for x in arrays) or np.any(arrays[0]<=0) or np.any(arrays[1]<=0) or np.any(arrays[2]<0):
        raise ValueError('finite market inputs, positive prices and nonnegative time required')
    if quadrature=='gauss_legendre':
        nodes,weights=_legendre(int(points));u=(b[1]-b[0])*(nodes+1)/2+b[0];w=weights*(b[1]-b[0])/2
    result=np.empty(arrays[0].shape)
    for idx in np.ndindex(result.shape):
        s,k,t,rate,div=(float(x[idx]) for x in arrays)
        if t==0:
            result[idx]=max((s-k)*(1 if kind=='call' else -1),0.);continue
        x=np.log(s/k)+(rate-div)*t
        def integrand(freq,shift):
            arg=freq-shift*1j
            cf=np.ones_like(arg,dtype=complex)
            for v0,kappa,theta,sigma,rho in p:
                if sigma==0:
                    iv=theta*t+(v0-theta)*(-np.expm1(-kappa*t))/kappa
                    cf*=np.exp(-.5*(arg*arg+1j*arg)*iv)
                else:
                    cf*=_heston_cf(arg,t,v0,kappa,theta,sigma,rho)
            return np.real(np.exp(1j*freq*x)*cf/(1j*freq))
        probabilities=[]
        for shift in (1,0):
            if quadrature=='quad':
                val=integrate.quad(lambda z:float(integrand(z,shift)),float(b[0]),float(b[1]),
                                   limit=int(limit),epsabs=epsabs,epsrel=epsrel)[0]
            else:
                val=float(w@integrand(u,shift))
            probabilities.append(.5+val/np.pi)
        p1,p2=probabilities;sp=s*np.exp(-div*t);kp=k*np.exp(-rate*t)
        result[idx]=sp*p1-kp*p2 if kind=='call' else kp*(1-p2)-sp*(1-p1)
    return float(result) if result.ndim==0 else result


def variance_replication(strikes, otm_prices, T, *, discount, endpoint):
    """Static log-contract variance replication on a nonuniform strike grid.

    Parameters
    ----------
    strikes : positive finite strictly increasing 1-D grid with at least two points.
    otm_prices : same-length nonnegative discounted OTM option prices. Caller
        chooses puts/calls and cleans the quotes using its supplied forward.
    T : strictly positive maturity in years.
    discount : strictly positive discount factor to maturity.
    endpoint : 'trapezoid' gives half adjacent gaps at endpoints; 'one_sided'
        gives full adjacent gaps. Interiors always use half the two-sided gap.

    Returns
    -------
    dict : variance, volatility, strike_widths and weights (width/K**2).
        Variance is 2*sum(weights*otm_prices)/(discount*T). No VIX forward
        correction or extrapolated tails are added. Finite-grid truncation is
        the caller's responsibility; endpoint convention is never guessed.

    Example
    -------
    >>> variance_replication([80,100,120],[1,8,2],.5,discount=.98,endpoint='one_sided')
    """
    k=np.asarray(strikes,dtype=float);p=np.asarray(otm_prices,dtype=float)
    if k.ndim!=1 or k.size<2 or p.shape!=k.shape or np.any(~np.isfinite(k)) or np.any(~np.isfinite(p)):
        raise ValueError('strikes and prices must be equal finite vectors')
    if np.any(k<=0) or np.any(np.diff(k)<=0) or np.any(p<0) or not 0<float(T)<np.inf or not 0<float(discount)<np.inf:
        raise ValueError('positive increasing strikes, nonnegative prices, positive time/discount required')
    if endpoint not in ('trapezoid','one_sided'): raise ValueError('invalid endpoint convention')
    width=np.r_[k[1]-k[0],(k[2:]-k[:-2])/2,k[-1]-k[-2]]
    if endpoint=='trapezoid': width[[0,-1]]*=.5
    weight=width/(k*k);variance=float(2*(weight@p)/(discount*T))
    return {'variance':variance,'volatility':float(np.sqrt(variance)),'strike_widths':width,'weights':weight}
