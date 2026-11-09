"""Short-rate lattice primitives with explicit cashflow and stopping conventions.

Moment matching follows R. Kohn's Continuous Time Finance, Section 6 (2004):
https://math.nyu.edu/~kohn/cont-time-finance/section6.pdf
"""
import numpy as np
from scipy.special import logsumexp

__all__ = ['hw_trinomial_tree', 'trinomial_rollback']


def hw_trinomial_tree(curve, a, sigma, *, dt, steps, j_max=None):
    """Hull-White trinomial lattice with Arrow-Debreu fitting of the initial curve.

    Parameters
    ----------
    curve : callable t -> positive discount factor, or object with .df(t) / .discount(t).
        Must support times 0 through (steps+1)*dt. The extra point calibrates
        the terminal node's short-rate shift for terminal exercise payoffs.
    a : finite nonnegative mean reversion per year.
    sigma : finite nonnegative absolute rate volatility per sqrt(year).
    dt : strictly positive uniform time step in years.
    steps : positive integer number of transitions.
    j_max : positive integer truncation index, or None for ceil(.184/(a*dt)),
        at least one; a=0 uses an untruncated tree. Invalid transition
        probabilities raise rather than silently changing the specified grid.

    Returns
    -------
    dict : times, nodes (integer j per layer), rates, alpha, state_prices,
        children (indices in next layer), probabilities, discount_factors,
        dx, dt, a, sigma, j_max. Transitions match Euler mean -a*x*dt and
        variance sigma**2*dt; dx=sigma*sqrt(3*dt). Curve fit is exact on this
        grid, but contingent-payoff discretization requires convergence checks.

    Example
    -------
    >>> tree = hw_trinomial_tree(lambda t:np.exp(-.03*np.asarray(t)),.1,.01,dt=.1,steps=20)
    >>> round(tree['state_prices'][-1].sum(),6)
    0.941765
    """
    if not np.all(np.isfinite([a,sigma,dt])) or a<0 or sigma<0 or dt<=0:
        raise ValueError('finite nonnegative a/sigma and positive dt required')
    if not isinstance(steps,(int,np.integer)) or steps<1:
        raise ValueError('steps must be a positive integer')
    if j_max is None:
        j_max=max(1,int(np.ceil(.184/(a*dt)))) if a else steps+1
    if not isinstance(j_max,(int,np.integer)) or j_max<1:
        raise ValueError('j_max must be a positive integer')
    discount=getattr(curve,'df',getattr(curve,'discount',curve))
    if not callable(discount): raise ValueError('curve must be callable or have discount(t)')
    times=np.arange(steps+1)*dt
    discounts=np.array([float(discount(t)) for t in np.arange(steps+2)*dt])
    if np.any(~np.isfinite(discounts)) or np.any(discounts<=0) or abs(discounts[0]-1)>1e-10:
        raise ValueError('curve must be positive and finite, with P(0)=1')
    dx=sigma*np.sqrt(3*dt)
    nodes=[np.arange(-min(n,j_max),min(n,j_max)+1) for n in range(steps+1)]
    state_prices=[np.array([1.])];alphas=[];rates=[];probabilities=[];children=[]
    for n,j in enumerate(nodes):
        q=state_prices[n];positive=q>0
        alpha=(logsumexp(np.log(q[positive])-j[positive]*dx*dt)-np.log(discounts[n+1]))/dt
        alphas.append(float(alpha));rates.append(alpha+j*dx)
        if n==steps: break
        # Solve the three moment equations at each branching type. Repeated
        # use of a fixed 3x3 matrix avoids separate sign-sensitive formulas.
        dest=np.empty((len(j),3),dtype=int);p=np.empty_like(dest,dtype=float)
        mean=-a*j*dt
        for offsets,mask in [(np.array([-1,0,1]),np.abs(j)<j_max),
                             (np.array([-2,-1,0]),j==j_max),
                             (np.array([0,1,2]),j==-j_max)]:
            if not np.any(mask): continue
            matrix=np.vstack([np.ones(3),offsets,offsets**2])
            moments=np.vstack([np.ones(mask.sum()),mean[mask],1/3+mean[mask]**2])
            p[mask]=np.linalg.solve(matrix,moments).T
            dest[mask]=j[mask,None]+offsets
        if np.any(p < -1e-13) or np.any(p>1+1e-13):
            raise ValueError('grid gives negative transition probabilities; refine dt or j_max')
        p=np.maximum(p,0);p/=p.sum(axis=1,keepdims=True)
        child=dest-nodes[n+1][0]
        if np.any(child<0) or np.any(child>=len(nodes[n+1])):
            raise ValueError('truncation is incompatible with next lattice layer')
        next_q=np.zeros(len(nodes[n+1]))
        np.add.at(next_q,child.ravel(),(q[:,None]*np.exp(-rates[n][:,None]*dt)*p).ravel())
        probabilities.append(p);children.append(child);state_prices.append(next_q)
    return {'times':times,'nodes':nodes,'rates':rates,'alpha':np.array(alphas),
            'state_prices':state_prices,'children':children,'probabilities':probabilities,
            'discount_factors':discounts[:-1],'dx':float(dx),'dt':float(dt),
            'a':float(a),'sigma':float(sigma),'j_max':int(j_max)}


def trinomial_rollback(tree, exercise_values, *, cashflows=None):
    """Trinomial lattice valuation with optional optimal stopping and dated cashflows.

    Parameters
    ----------
    tree : result of hw_trinomial_tree.
    exercise_values : mapping integer step -> finite nonnegative scalar or
        vector matching that layer's nodes. Empty mapping disables exercise.
        Caller builds the payoff from current-layer rates and remaining
        contract cashflows; date-to-grid alignment is the caller's choice.
    cashflows : mapping step -> finite scalar or node vector, default empty.
        At each step V=cashflow+max(exercise,discounted continuation).
        Thus cashflows at the exercise date are paid before the stopping
        decision and are received in either branch. Do not also put them
        inside exercise_values. No terminal redemption is inferred.

    Returns
    -------
    dict : value at time zero, values per layer, continuation per layer, and
        exercised boolean masks for specified steps. A positive payoff must
        strictly exceed continuation to set exercised=True; ties continue.

    Example
    -------
    >>> t=hw_trinomial_tree(lambda x:np.exp(-.03*np.asarray(x)),.1,.01,dt=.1,steps=10)
    >>> round(trinomial_rollback(t,{},cashflows={10:1.})['value'],6)
    0.970446
    """
    count=len(tree['nodes']);cashflows={} if cashflows is None else cashflows
    def check(mapping,nonnegative):
        out={}
        for step,v in mapping.items():
            if not isinstance(step,(int,np.integer)) or not 0<=step<count:
                raise ValueError('cashflow/exercise step is outside tree')
            arr=np.asarray(v,dtype=float)
            if arr.ndim>1 or (arr.ndim==1 and len(arr)!=len(tree['nodes'][step])):
                raise ValueError('cashflow/exercise vector must match layer nodes')
            if np.any(~np.isfinite(arr)) or (nonnegative and np.any(arr<0)):
                raise ValueError('cashflows must be finite; exercise values nonnegative')
            out[step]=np.broadcast_to(arr,(len(tree['nodes'][step]),)).copy()
        return out
    exercises=check(exercise_values,True);cash=check(cashflows,False)
    values=[None]*count;continuations=[None]*count;masks={}
    for n in range(count-1,-1,-1):
        if n==count-1: cont=np.zeros(len(tree['nodes'][n]))
        else:
            expected=np.sum(tree['probabilities'][n]*values[n+1][tree['children'][n]],axis=1)
            cont=np.exp(-tree['rates'][n]*tree['dt'])*expected
        continuations[n]=cont
        if n in exercises:
            masks[n]=(exercises[n]>cont)&(exercises[n]>0)
            cont=np.maximum(exercises[n],cont)
        values[n]=cont+cash.get(n,0.)
    return {'value':float(values[0][0]),'values':values,'continuation':continuations,'exercised':masks}
