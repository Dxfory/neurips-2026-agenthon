"""Weighted empirical tails, observed GPD information and smooth loss mixtures.

Losses are positive for losses; alpha is a confidence level, not tail mass.
Inputs contain no dates or asset labels. No sampling or data loading is performed.
"""
import numpy as np
from scipy import integrate, optimize, stats

__all__ = ['weighted_var_es', 'gpd_observed_information', 'normal_mixture_var_es', 'kde_var_es']


def _vector(x, name):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)==0 or np.any(~np.isfinite(x)):
        raise ValueError(f'{name} must be a nonempty finite vector')
    return x


def _alpha(alpha):
    a=float(alpha)
    if not 0<a<1:
        raise ValueError('alpha must be a confidence level in (0,1)')
    return a


def _weights(weights, n):
    w=_vector(weights,'weights')
    if len(w)!=n or np.any(w<0) or not 0<float(w.sum())<np.inf:
        raise ValueError('weights must match observations, be nonnegative and have finite positive sum')
    return w/w.sum()


def weighted_var_es(losses, weights, alpha, *, quantile, es_rule):
    """Weighted historical VaR and expected shortfall with explicit tail conventions.

    Parameters
    ----------
    losses : finite 1-D loss sample, positive means loss.
    weights : same-length nonnegative observation weights, normalized internally.
        For age weighting, caller supplies decay**age with newest age zero.
    alpha : confidence level in (0,1).
    quantile : 'inverted_cdf' smallest loss whose cumulative weight reaches alpha,
        or 'linear_cdf' interpolation through sorted losses and cumulative weights.
    es_rule : 'fractional' integrates the upper 1-alpha mass of the empirical
        quantile function; 'ge' or 'gt' averages observations >= or > reported
        VaR with their weights, falling back to VaR if the selected mass is zero.

    Returns
    -------
    dict : var and es. Zero-weight points are removed before quantile estimation.
        Fractional ES uses the empirical distribution even with linear_cdf VaR.

    Example
    -------
    >>> weighted_var_es([1,3,8],[.2,.5,.3],.6,quantile='inverted_cdf',es_rule='fractional')
    {'var': 3.0, 'es': 6.75}
    """
    x=_vector(losses,'losses');w=_weights(weights,len(x));a=_alpha(alpha)
    if quantile not in ('inverted_cdf','linear_cdf') or es_rule not in ('fractional','ge','gt'):
        raise ValueError('unsupported quantile or es_rule')
    keep=w>0;x=x[keep];w=w[keep];order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    c=np.cumsum(w);c[-1]=1.
    v=float(x[np.searchsorted(c,a,side='left')]) if quantile=='inverted_cdf' else float(np.interp(a,c,x))
    if es_rule=='fractional':
        tail=np.maximum(c-a,0)-np.maximum(np.r_[0.,c[:-1]]-a,0)
        es=float(tail@x/(1-a))
    else:
        use=x>=v if es_rule=='ge' else x>v
        es=float(w[use]@x[use]/w[use].sum()) if w[use].sum()>0 else v
    return {'var':v,'es':es}


def gpd_observed_information(excesses, xi, beta, *, rel_step, max_halvings):
    """Observed information for a generalized Pareto likelihood at supplied parameters.

    Parameters
    ----------
    excesses : finite nonnegative 1-D threshold excesses (not raw losses).
    xi, beta : fitted GPD shape and strictly positive scale, location fixed at zero.
    rel_step : positive central-difference step relative to max(1,abs(xi))
        for shape and beta for scale; e.g. 1e-4. Choose/check numerical stability.
    max_halvings : nonnegative integer bound for shrinking steps until all
        evaluation points satisfy 1+xi*y/beta>0 and beta>0.

    Returns
    -------
    dict : information (2x2 in xi,beta order), covariance (its inverse), se_xi,
        se_beta and steps. Raises if support cannot be maintained or observed
        information is not positive definite; no expected-information fallback.

    Example
    -------
    >>> result = gpd_observed_information(excesses, .2, .01, rel_step=1e-4, max_halvings=10)
    """
    y=_vector(excesses,'excesses');xi=float(xi);beta=float(beta);step=float(rel_step)
    if np.any(y<0) or not np.all(np.isfinite([xi,beta,step])) or beta<=0 or step<=0:
        raise ValueError('nonnegative excesses, finite shape and positive scale/step required')
    if isinstance(max_halvings,bool) or int(max_halvings)!=max_halvings or max_halvings<0:
        raise ValueError('max_halvings must be a nonnegative integer')
    def nll(shape,scale):
        if scale<=0: return np.inf
        z=y/scale;u=shape*z
        if np.any(u<=-1): return np.inf
        if abs(shape)<1e-5 and np.max(np.abs(u))<.01:
            # log(1+xi*z)/xi has a continuous, differentiable extension at zero.
            g=z-shape*z*z/2+shape**2*z**3/3-shape**3*z**4/4+shape**4*z**5/5
            return float(len(y)*np.log(scale)+(np.log1p(u)+g).sum())
        return float(len(y)*np.log(scale)+(1+1/shape)*np.log1p(u).sum())
    h=np.array([step*max(1.,abs(xi)),step*beta]);f=nll(xi,beta)
    if not np.isfinite(f): raise ValueError('fitted parameters violate GPD support')
    for _ in range(int(max_halvings)+1):
        hx,hb=h
        fs=np.array([nll(xi+hx,beta),nll(xi-hx,beta),nll(xi,beta+hb),nll(xi,beta-hb),
                     nll(xi+hx,beta+hb),nll(xi+hx,beta-hb),nll(xi-hx,beta+hb),nll(xi-hx,beta-hb)])
        if np.all(np.isfinite(fs)): break
        h=h/2
    else: raise ValueError('finite Hessian stencil not found within GPD support')
    cross=(fs[4]-fs[5]-fs[6]+fs[7])/(4*hx*hb)
    info=np.array([[(fs[0]-2*f+fs[1])/hx**2,cross],[cross,(fs[2]-2*f+fs[3])/hb**2]])
    if np.any(~np.isfinite(info)) or np.min(np.linalg.eigvalsh(info))<=0:
        raise ValueError('observed information is not positive definite')
    cov=np.linalg.inv(info)
    return {'information':info,'covariance':cov,'se_xi':float(np.sqrt(cov[0,0])),
            'se_beta':float(np.sqrt(cov[1,1])),'steps':h}


def normal_mixture_var_es(weights, means, scales, alpha, *, bounds, es_method):
    """Gaussian mixture loss VaR and expected shortfall.

    Parameters
    ----------
    weights, means, scales : same-length mixture probabilities, component means
        and strictly positive standard deviations (not variances).
    alpha : confidence level in (0,1).
    bounds : finite increasing pair bracketing the mixture quantile for Brent.
    es_method : 'analytic' uses Normal truncated moments; 'quad' integrates
        x times the mixture density from VaR to infinity using scipy.integrate.quad.

    Returns
    -------
    dict : var and es in loss units; weights are normalized internally.

    Example
    -------
    >>> normal_mixture_var_es([.8,.2],[0.,1.],[1.,2.],.99,bounds=(-10.,20.),es_method='quad')
    """
    mu=_vector(means,'means');sd=_vector(scales,'scales');w=_weights(weights,len(mu));a=_alpha(alpha)
    if len(sd)!=len(mu) or np.any(sd<=0) or es_method not in ('analytic','quad'):
        raise ValueError('positive component scales and valid es_method required')
    bracket=_vector(bounds,'bounds')
    if len(bracket)!=2 or bracket[0]>=bracket[1]: raise ValueError('bounds must be an increasing pair')
    v=float(optimize.brentq(lambda x:float(w@stats.norm.cdf((x-mu)/sd))-a,*bracket))
    if es_method=='analytic':
        z=(v-mu)/sd;tail=float(w@(mu*stats.norm.sf(z)+sd*stats.norm.pdf(z)))
    else:
        tail=integrate.quad(lambda x:x*float(w@(stats.norm.pdf((x-mu)/sd)/sd)),v,np.inf,epsabs=1e-10)[0]
    return {'var':v,'es':float(tail/(1-a))}


def kde_var_es(losses, alpha, *, bw_method, grid, epsabs):
    """Gaussian KDE loss VaR by CDF-grid interpolation and ES by tail quadrature.

    Parameters
    ----------
    losses : nonconstant finite 1-D loss sample, positive means loss.
    alpha : confidence level in (0,1).
    bw_method : scipy.stats.gaussian_kde bandwidth selector, e.g. 'silverman'.
    grid : finite strictly increasing quantile search grid. Its CDF endpoints
        must bracket alpha; no implicit extension, extrapolation or root finding.
    epsabs : positive absolute tolerance for scipy.integrate.quad on x*kde(x).

    Returns
    -------
    dict : var, es, bandwidth (standard deviation), tail_integral_error. CDF
        evaluations use gaussian_kde.integrate_box_1d; ES divides by 1-alpha.

    Example
    -------
    >>> result = kde_var_es(losses, .99, bw_method='silverman', grid=grid, epsabs=1e-8)
    """
    x=_vector(losses,'losses');a=_alpha(alpha);g=_vector(grid,'grid');epsabs=float(epsabs)
    if len(x)<2 or np.var(x)<=0 or len(g)<2 or np.any(np.diff(g)<=0) or not 0<epsabs<np.inf:
        raise ValueError('nonconstant sample, increasing grid and positive tolerance required')
    kde=stats.gaussian_kde(x,bw_method=bw_method)
    cdf=np.array([kde.integrate_box_1d(-np.inf,float(v)) for v in g])
    if not cdf[0]<=a<=cdf[-1]: raise ValueError('grid CDF endpoints do not bracket alpha')
    v=float(np.interp(a,cdf,g))
    tail,error=integrate.quad(lambda z:z*float(kde(z)[0]),v,np.inf,epsabs=epsabs)
    return {'var':v,'es':float(tail/(1-a)),'bandwidth':float(np.sqrt(kde.covariance[0,0])),
            'tail_integral_error':float(error)}
