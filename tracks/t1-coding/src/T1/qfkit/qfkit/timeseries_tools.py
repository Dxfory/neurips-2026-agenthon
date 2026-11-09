"""Time-series alignment and correlation diagnostics with explicit conventions."""
import numpy as np
from scipy.stats import norm

__all__=['lag_correlations','fisher_correlation_difference']


def lag_correlations(a, b, *, lags, min_pairs):
    """Lead-lag Pearson correlations, with positive lag meaning the first series leads.

    Parameters
    ----------
    a, b : equal-length 1-D chronological numeric vectors, already calendar-aligned.
        Nonfinite pairs are excluded separately at each lag; missing timestamps
        must be aligned before calling, never compressed out of the time index.
    lags : 1-D vector of integer lags, in periods (any input order preserved).
    min_pairs : integer >=2; fewer finite overlapping observations return NaN.

    Returns
    -------
    dict : lags, correlation and n_pairs vectors. Uses numpy.corrcoef for
        corr(a[t],b[t+k]); constant samples and lags outside the sample have
        NaN correlation. No selection, significance or absolute-value ranking
        is performed. Positive and negative peaks are caller-defined statistics.

    Example
    -------
    >>> lag_correlations([1,3,2,4],[0,1,3,2],lags=[1],min_pairs=3)['correlation']
    array([1.])
    """
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);lags=np.asarray(lags)
    if a.ndim!=1 or b.shape!=a.shape or lags.ndim!=1 or not np.issubdtype(lags.dtype,np.integer):
        raise ValueError('aligned 1-D series and integer lag vector required')
    if not isinstance(min_pairs,(int,np.integer)) or min_pairs<2:
        raise ValueError('min_pairs must be an integer >=2')
    corr=np.full(len(lags),np.nan);counts=np.zeros(len(lags),dtype=int);n=len(a)
    for i,k in enumerate(lags):
        k=int(k)
        if abs(k)>=n: continue
        x=a[:n-k] if k>=0 else a[-k:]
        y=b[k:] if k>=0 else b[:n+k]
        valid=np.isfinite(x)&np.isfinite(y);x=x[valid];y=y[valid];counts[i]=len(x)
        if len(x)>=min_pairs and np.ptp(x)>0 and np.ptp(y)>0:
            corr[i]=np.corrcoef(x,y)[0,1]
    return {'lags':lags.copy(),'correlation':corr,'n_pairs':counts}


def fisher_correlation_difference(rho_a, rho_b, *, n_a, n_b, clip, level):
    """Independent-sample Fisher z approximation for a difference of correlations.

    Parameters
    ----------
    rho_a, rho_b : finite correlations in [-1,1], broadcastable with n_a/n_b.
    n_a, n_b : sample sizes strictly greater than 3, chosen by caller.
    clip : positive number <1; correlations are clipped to [-clip,clip] before atanh.
    level : two-sided confidence level in (0,1).

    Returns
    -------
    dict : delta_z, se=sqrt(1/(n_a-3)+1/(n_b-3)), z, pvalue,
        critical_z and significant. This approximation assumes independent
        estimates; it does not correct shared observations, autocorrelation or
        searching across lags. With selected/dependent peaks it is a diagnostic
        only, not a calibrated test of predictive ability.

    Example
    -------
    >>> fisher_correlation_difference(.3,.3,n_a=80,n_b=80,clip=.999,level=.95)['z']
    0.0
    """
    ra,rb,na,nb=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in (rho_a,rho_b,n_a,n_b)])
    if any(np.any(~np.isfinite(x)) for x in (ra,rb,na,nb)) or np.any(np.abs(ra)>1) or np.any(np.abs(rb)>1):
        raise ValueError('finite correlations in [-1,1] and sample sizes required')
    if np.any(na<=3) or np.any(nb<=3) or not 0<clip<1 or not 0<level<1:
        raise ValueError('sample sizes >3 and clip/level strictly in (0,1) required')
    delta=np.arctanh(np.clip(ra,-clip,clip))-np.arctanh(np.clip(rb,-clip,clip))
    se=np.sqrt(1/(na-3)+1/(nb-3));z=delta/se;critical=float(norm.ppf((1+level)/2))
    return {'delta_z':delta,'se':se,'z':z,'pvalue':2*norm.sf(np.abs(z)),
            'critical_z':critical,'significant':np.abs(z)>critical}
