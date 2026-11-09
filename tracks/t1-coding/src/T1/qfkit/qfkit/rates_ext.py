"""Generic dated schedules, discount/projection curves and cashflow risk.

Rates are decimals, times are years, FX is quote currency per base currency.
Calendar accrual dates and adjusted payment dates are separate. Cubic interpolation
is not-a-knot on integrated continuous zero rate g(t)=-log D(t), including g(0)=0.
No task input/output, security selection, market defaults or inferred fixings.
"""
from __future__ import annotations

import calendar
import math

import numpy as np
import pandas as pd
from scipy import interpolate, optimize

from .curves import (_par_instrument_flows, _parse_freq, _short_end_rule, bond_cashflows, df_from_rate,
                     ns_yield, rate_from_df, year_fraction)

__all__ = [
    'timestamp_year_fraction', 'dated_year_fraction', 'dated_schedule',
    'DiscountCurve', 'bootstrap_par_curve', 'bootstrap_projection_curve',
    'floating_leg_cashflows', 'dual_curve_swap', 'mtm_principal_cashflows',
    'price_cashflows', 'curve_cashflow_risk', 'cashflow_ytm', 'z_spread',
    'key_rate_risk', 'quote_key_rate_risk', 'accrued_interest',
    'deterministic_call_price_cap', 'fx_forward', 'fx_implied_basis',
    'fit_parametric_curve',
]


def _ret(value):
    a = np.asarray(value, dtype=float)
    return float(a) if a.ndim == 0 else a


def _vector(value, name, *, nonnegative=False, positive=False):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be a nonempty finite 1-D array')
    if nonnegative and np.any(a < 0) or positive and np.any(a <= 0):
        raise ValueError(f'{name} outside allowed range')
    return a


def _positive(value, name):
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return v


def _scalar(value, name):
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f'{name} must be finite')
    return v


def _df(curve, times):
    # Both qfkit ZeroCurve and DiscountCurve, or an explicit DF callable.
    d = np.asarray(curve.df(times) if hasattr(curve, 'df') else curve(times), dtype=float)
    d = np.broadcast_to(d, np.asarray(times).shape)
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError('curve returned nonpositive or nonfinite discount factors')
    return d


def _cashflows(times, amounts):
    t = _vector(times, 'times', nonnegative=True)
    a = _vector(amounts, 'amounts')
    if t.shape != a.shape:
        raise ValueError('times and amounts must have equal shape')
    return t, a


def _root(function, bracket, xtol):
    lo, hi = map(float, bracket)
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError('bracket must be two finite increasing bounds')
    return float(optimize.brentq(function, lo, hi, xtol=_positive(xtol, 'xtol')))


def timestamp_year_fraction(start, end, *, days_per_year):
    """Elapsed timestamp time in years without truncating intraday seconds.

    Parameters
    ----------
    start, end : scalar or broadcastable arrays of datetime-like values.
        Aware values are converted to UTC; naive values are treated as UTC.
    days_per_year : positive float, explicit denominator (e.g. 365 or 365.25).

    Returns
    -------
    float or ndarray : signed elapsed seconds / (86400 * days_per_year).

    Example
    -------
    >>> timestamp_year_fraction('2024-01-01T00:00Z', '2024-01-01T12:00Z', days_per_year=365)
    0.0013698630136986301
    """
    denom = 86400 * _positive(days_per_year, 'days_per_year')
    a, b = np.broadcast_arrays(np.asarray(start, dtype=object), np.asarray(end, dtype=object))
    result = np.empty(a.shape, dtype=float)
    for index in np.ndindex(a.shape):
        s, e = pd.to_datetime(a[index], utc=True), pd.to_datetime(b[index], utc=True)
        if pd.isna(s) or pd.isna(e):
            raise ValueError('timestamps must not contain NaT')
        result[index] = (e - s).total_seconds() / denom
    return _ret(result)


def dated_year_fraction(start, end, *, basis):
    """Calendar date accrual fractions, separate from intraday elapsed time.

    Parameters
    ----------
    start, end : scalar or broadcastable dates. Intraday parts are discarded.
    basis : 'act/360', 'act/365f', 'act/365.25', 'act/act_isda',
        '30e/360' or '30/360_us'. US includes end-of-February adjustment:
        start Feb EOM becomes 30; end Feb EOM becomes 30 when both are Feb EOM;
        end 31 becomes 30 when adjusted start >= 30. This differs from the
        legacy curves.year_fraction US variant which omits February adjustment.

    Returns
    -------
    float or ndarray : signed calendar accrual fractions.

    Example
    -------
    >>> dated_year_fraction('2024-02-29', '2024-03-31', basis='30/360_us')
    0.08333333333333333
    """
    key = str(basis).lower()
    if key == 'act/365.25':
        s = pd.to_datetime(np.asarray(start).ravel()).normalize()
        e = pd.to_datetime(np.asarray(end).ravel()).normalize()
        return timestamp_year_fraction(np.asarray(s).reshape(np.asarray(start).shape),
                                       np.asarray(e).reshape(np.asarray(end).shape), days_per_year=365.25)
    if key != '30/360_us':
        return year_fraction(start, end, basis=basis)
    a, b = np.broadcast_arrays(np.asarray(start, dtype=object), np.asarray(end, dtype=object))
    values = np.empty(a.shape)
    for index in np.ndindex(a.shape):
        s, e = pd.Timestamp(a[index]).normalize(), pd.Timestamp(b[index]).normalize()
        if pd.isna(s) or pd.isna(e):
            raise ValueError('dates must not contain NaT')
        sign = 1
        if e < s:
            s, e, sign = e, s, -1
        sf = s.month == 2 and s.day == calendar.monthrange(s.year, 2)[1]
        ef = e.month == 2 and e.day == calendar.monthrange(e.year, 2)[1]
        d1 = 30 if s.day == 31 or sf else s.day
        d2 = 30 if (ef and sf) or (e.day == 31 and d1 >= 30) else e.day
        values[index] = sign * (360*(e.year-s.year) + 30*(e.month-s.month) + d2-d1)/360
    return _ret(values)


def dated_schedule(start, end, *, months, direction, stub, end_of_month, business_day, holidays):
    """Build calendar coupon/reset periods; adjust only payment dates.

    Parameters
    ----------
    start, end : date-like, start < end, interpreted as local calendar dates.
    months : positive integer tenor in months.
    direction : 'backward' (anchor end) or 'forward' (anchor start).
    stub : 'short', 'long' (merge irregular period with neighbor), or 'none'
        (reject irregular schedule). The stub is at the unanchored end.
    end_of_month : bool; preserve EOM only when the anchor itself is EOM.
    business_day : 'unadjusted', 'following', 'modified_following', 'preceding'.
    holidays : explicit date collection; weekend is Saturday/Sunday.

    Returns
    -------
    pandas.DataFrame : accrual_start/end (unadjusted), payment_date, is_stub.
        Accrual fractions are deliberately not inferred; use dated_year_fraction.

    Example
    -------
    >>> dated_schedule('2024-01-31', '2025-01-31', months=6, direction='backward', stub='none', end_of_month=True, business_day='unadjusted', holidays=[]).shape
    (2, 4)
    """
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if pd.isna(s) or pd.isna(e) or s >= e or s.tzinfo or e.tzinfo:
        raise ValueError('start/end must be increasing timezone-naive calendar dates')
    if isinstance(months, bool) or int(months) != months or months <= 0:
        raise ValueError('months must be a positive integer')
    if direction not in ('backward', 'forward') or stub not in ('short', 'long', 'none'):
        raise ValueError('invalid direction or stub')
    if business_day not in ('unadjusted', 'following', 'modified_following', 'preceding'):
        raise ValueError('invalid business_day')
    if not isinstance(end_of_month, (bool, np.bool_)):
        raise ValueError('end_of_month must be bool')
    anchor = e if direction == 'backward' else s
    step = -int(months) if direction == 'backward' else int(months)
    at_eom = bool(end_of_month and anchor.is_month_end)
    points = [anchor]
    j = 1
    while True:
        date = anchor + pd.DateOffset(months=step*j)
        if at_eom:
            date += pd.offsets.MonthEnd(0)
        if direction == 'backward' and date <= s or direction == 'forward' and date >= e:
            boundary = s if direction == 'backward' else e
            irregular = date != boundary
            points.append(boundary)
            break
        points.append(date)
        j += 1
    if direction == 'backward':
        points.reverse()
    if irregular and stub == 'none':
        raise ValueError('schedule has a stub with stub=none')
    if irregular and stub == 'long' and len(points) > 2:
        del points[1 if direction == 'backward' else -2]
    holidays = {pd.Timestamp(d).normalize() for d in holidays}
    def business(date):
        return date.weekday() < 5 and date not in holidays
    def adjust(date):
        if business_day == 'unadjusted' or business(date):
            return date
        step = -1 if business_day == 'preceding' else 1
        ans = date
        while not business(ans):
            ans += pd.Timedelta(days=step)
        if business_day == 'modified_following' and ans.month != date.month:
            ans = date
            while not business(ans):
                ans -= pd.Timedelta(days=1)
        return ans
    flags = np.zeros(len(points)-1, dtype=bool)
    if irregular:
        flags[0 if direction == 'backward' else -1] = True
    return pd.DataFrame({'accrual_start':points[:-1], 'accrual_end':points[1:],
                         'payment_date':[adjust(d) for d in points[1:]], 'is_stub':flags})


class DiscountCurve:
    """Discount factor curve with explicit interpolation and extrapolation.

    Parameters
    ----------
    times : strictly increasing positive node times. An explicit zero node is
        also accepted only when its DF is 1 (zero-rate input may use any finite zero).
    values : zero rates or discount factors, same length as times.
    input_type : 'zero' or 'df'. comp : explicit zero-rate compounding convention.
    interp : 'linear_zero', 'log_df', 'cubic_rt' (not-a-knot cubic on g=-log D).
    extrapolate : 'flat_zero', 'flat_forward' or 'raise'. Between 0 and the
        first node, interpolation includes g(0)=0; linear_zero uses first zero.

    Returns
    -------
    DiscountCurve : df(t), zero(t,comp=...), shifted(continuous_zero_bumps).
        Nodes need not have decreasing discount factors: negative rates allowed.

    Example
    -------
    >>> DiscountCurve([1,2],[.03,.03],input_type='zero',comp='continuous',interp='cubic_rt',extrapolate='flat_zero').df(0)
    1.0
    """
    def __init__(self, times, values, *, input_type, comp, interp, extrapolate):
        t = _vector(times, 'times', nonnegative=True)
        v = _vector(values, 'values')
        if t.shape != v.shape or np.any(np.diff(t) <= 0):
            raise ValueError('node times must increase strictly and match values')
        if input_type not in ('zero', 'df') or interp not in ('linear_zero', 'log_df', 'cubic_rt'):
            raise ValueError('invalid input_type or interp')
        if extrapolate not in ('flat_zero', 'flat_forward', 'raise'):
            raise ValueError('invalid extrapolate')
        # Validate comp even for DF input.
        df_from_rate(0, 1, comp=comp)
        d = np.asarray(df_from_rate(v, t, comp=comp) if input_type == 'zero' else v)
        if np.any(d <= 0) or np.any(~np.isfinite(d)):
            raise ValueError('discount factors must be positive finite')
        if t[0] == 0:
            if abs(d[0]-1) > 1e-14:
                raise ValueError('D(0) must equal 1')
            t, d = t[1:], d[1:]
        if not t.size:
            raise ValueError('need a positive-time node')
        self.times, self.discounts = t.copy(), d.copy()
        self.zeros = -np.log(d)/t
        self.interp, self.extrapolate = interp, extrapolate
        self._t = np.r_[0., t]; self._g = np.r_[0., -np.log(d)]
        self._spline = interpolate.CubicSpline(self._t, self._g, bc_type='not-a-knot') if interp == 'cubic_rt' else None

    def df(self, times):
        """Evaluate positive discount factors for finite nonnegative year times."""
        t = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0):
            raise ValueError('evaluation times must be finite and nonnegative')
        if self.extrapolate == 'raise' and np.any(t > self.times[-1]):
            raise ValueError('time beyond curve last node')
        if self.interp == 'linear_zero':
            g = np.interp(t, self.times, self.zeros)*t
        elif self.interp == 'log_df':
            g = np.interp(t, self._t, self._g)
        else:
            g = self._spline(t)
        outside = t > self.times[-1]
        if self.extrapolate == 'flat_zero':
            g = np.where(outside, self.zeros[-1]*t, g)
        elif self.extrapolate == 'flat_forward':
            slope = (self._spline(self.times[-1], 1) if self._spline is not None
                     else ((self._g[-1]-self._g[-2])/(self._t[-1]-self._t[-2]) if self.interp == 'log_df'
                           else (self.zeros[-1]+self.times[-1]*(self.zeros[-1]-self.zeros[-2])/(self.times[-1]-self.times[-2]) if self.times.size > 1 else self.zeros[-1])))
            g = np.where(outside, self._g[-1]+slope*(t-self.times[-1]), g)
        with np.errstate(over='raise', under='raise'):
            return _ret(np.exp(-g))

    __call__ = df

    def zero(self, times, *, comp):
        """Zero rates in explicit comp; t=0 uses first positive-node zero."""
        t = np.asarray(times, dtype=float)
        z = np.asarray(rate_from_df(self.df(np.where(t == 0, self.times[0], t)),
                                    np.where(t == 0, self.times[0], t), comp=comp))
        return _ret(z)

    def shifted(self, bumps):
        """New curve with supplied continuous zero-rate node bumps (decimals)."""
        b = np.broadcast_to(np.asarray(bumps, dtype=float), self.zeros.shape)
        if not np.isfinite(b).all():
            raise ValueError('bumps must be finite and match nodes')
        return DiscountCurve(self.times, self.zeros+b, input_type='zero', comp='continuous',
                             interp=self.interp, extrapolate=self.extrapolate)


def _calibrate(times, residuals, initial, *, interp, extrapolate, bracket, xtol):
    bounds = np.asarray(bracket, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('bracket must be finite increasing continuous-zero bounds')
    xtol = _positive(xtol, 'xtol')
    def make(z):
        return DiscountCurve(times, z, input_type='zero', comp='continuous', interp=interp, extrapolate=extrapolate)
    def fun(z):
        ans = np.asarray(residuals(make(z)), dtype=float)
        if not np.isfinite(ans).all():
            raise ValueError('nonfinite calibration residual')
        return ans
    tol = max(xtol, 10*np.finfo(float).eps)
    result = optimize.least_squares(fun, np.clip(initial, bounds[0]+1e-10, bounds[1]-1e-10),
                                   bounds=(bounds[0],bounds[1]), ftol=tol, xtol=tol, gtol=tol,
                                   max_nfev=5000)
    error = float(np.max(np.abs(fun(result.x))))
    if not result.success or error > max(1e-10, 10*xtol):
        raise ValueError(f'curve calibration did not reprice inputs: max residual={error:.3g}; {result.message}')
    return make(result.x)


def bootstrap_par_curve(maturities, rates, *, frequency, interp, extrapolate, bracket, xtol, short_end='full_coupon'):
    """Calibrate a zero curve to unit-price fixed-coupon par instruments.

    A par instrument maturing within one coupon period (T < 1/frequency, rate != 0) is one
    payment at T. By default (short_end='full_coupon', the bond's own cash flows) it pays a full coupon; short_end='compounded' makes it a zero-coupon instrument at the
    quoted rate on the bond's compounding basis, (1 + c/f)^(f T); 'simple' reads the rate as
    a money-market yield, 1 + c T; 'full_coupon' pays a full coupon, 1 + c/f (the
    bond_cashflows reading).

    Parameters
    ----------
    maturities, rates : increasing positive times and decimal par coupons.
    frequency : explicit coupon payments/year; cashflows count backward from
        maturity, and a short first period of a longer bond pays a full coupon (the
        seasoned-bond convention of bond_cashflows).
    interp, extrapolate : DiscountCurve interpolation/extrapolation.
    bracket : two bounds on continuous node zero rates, negative rates allowed.
    xtol : positive calibration stopping tolerance; all par PV residuals checked.
    short_end : 'full_coupon' (default), 'compounded' or 'simple'; the rule for an instrument
        maturing within one coupon period (see above).

    Returns
    -------
    DiscountCurve : simultaneous exact repricing on the final global interpolant.
        A cubic spline changes earlier coupon discounts when later nodes are added;
        simultaneous calibration avoids a false sequential-bootstrap repricing claim.

    Example
    -------
    >>> bootstrap_par_curve([1,2],[.03,.035],frequency=2,interp='log_df',extrapolate='flat_zero',bracket=(-.5,1),xtol=1e-12).df(0)
    1.0
    """
    t = _vector(maturities, 'maturities', positive=True); r = _vector(rates, 'rates')
    if t.shape != r.shape or np.any(np.diff(t) <= 0):
        raise ValueError('maturities must increase strictly and match rates')
    rule = _short_end_rule(short_end)
    flows = [_par_instrument_flows(T, coupon, _parse_freq(frequency), 1., rule) for T,coupon in zip(t,r)]
    residual = lambda c: [np.dot(a, _df(c, u))-1 for u,a in flows]
    return _calibrate(t, residual, r, interp=interp, extrapolate=extrapolate, bracket=bracket, xtol=xtol)


def _float_inputs(starts, ends, pay_times, accruals, notionals, fixings):
    s = _vector(starts,'starts'); e = _vector(ends,'ends',positive=True)
    t = _vector(pay_times,'pay_times',positive=True); a = _vector(accruals,'accruals',positive=True)
    if any(v.shape != s.shape for v in (e,t,a)) or np.any(e <= s) or np.any(t < e):
        raise ValueError('period arrays must match with start < end <= payment')
    n = np.broadcast_to(np.asarray(notionals,dtype=float),s.shape)
    if not np.isfinite(n).all() or np.any(n < 0):
        raise ValueError('notional must be finite nonnegative')
    f = np.full(s.shape,np.nan) if fixings is None else np.broadcast_to(np.asarray(fixings,dtype=float),s.shape)
    if np.isinf(f).any():
        raise ValueError('fixings must be finite or NaN for projected coupons')
    if np.any((s < 0) & ~np.isfinite(f)):
        raise ValueError('historical starts require explicit finite fixings')
    return s,e,t,a,n,f


def floating_leg_cashflows(starts, ends, pay_times, accruals, *, projection_curve, notional, spread, fixings):
    """Interest-only floating leg using explicit known or projected fixings.

    Parameters
    ----------
    starts, ends, pay_times, accruals : aligned period/reset/end/payment times and
        accrual fractions. Payment may occur after accrual end. A historical
        start < 0 is allowed only with a finite known fixing; end/payment > 0.
    projection_curve : DF curve; forward=(P(start)/P(end)-1)/accrual.
    notional : nonnegative scalar or per-period notionals. spread : decimal.
    fixings : explicit rates array; finite entries override projection, NaN
        projects; None projects all. Caller filters settled cashflows and fixes
        valuation-date coupons. No implicit historical-fixing lookup occurs.

    Returns
    -------
    dict : times, amounts, rates (before spread), notionals. No principal exchanges.

    Example
    -------
    >>> floating_leg_cashflows([0],[1],[1],[1],projection_curve=lambda t: np.ones_like(t),notional=100,spread=.01,fixings=[.02])['amounts'].tolist()
    [3.0]
    """
    s,e,t,a,n,known = _float_inputs(starts,ends,pay_times,accruals,notional,fixings)
    rates = known.copy()
    project = ~np.isfinite(known)
    if np.any(project):
        rates[project] = (_df(projection_curve,s[project])/_df(projection_curve,e[project])-1)/a[project]
    return {'times':t.copy(),'amounts':n*a*(rates+_scalar(spread,'spread')),
            'rates':rates,'notionals':n.copy()}


def dual_curve_swap(fixed_times, fixed_accruals, float_starts, float_ends, float_times, float_accruals, *, discount_curve, projection_curve, fixed_rate, notional, spread, pay_fixed, fixings):
    """Value a fixed/floating swap with separate discount and projection curves.

    Parameters
    ----------
    fixed_times/accruals : explicit fixed payment times/accrual fractions.
    float_starts/ends/times/accruals : explicit floating reset/end/payment arrays.
    discount_curve, projection_curve : DF curves with independently defined conventions.
    fixed_rate, spread : decimal fixed rate and all-in floating spread.
    notional : positive scalar, constant across both legs. pay_fixed : bool.
    fixings : known floating rates or NaN projection; None projects all.

    Returns
    -------
    dict : fixed_pv, float_pv (unsigned leg values, may be negative if coupons
        are negative), net_pv (float-fixed for payer), annuity and par_rate.
        Interest legs only; principal exchanges are separate cashflows.

    Example
    -------
    >>> dual_curve_swap([1],[1],[0],[1],[1],[1],discount_curve=lambda t: np.ones_like(t),projection_curve=lambda t: np.ones_like(t),fixed_rate=.02,notional=100,spread=0,pay_fixed=True,fixings=[.03])['net_pv']
    1.0
    """
    t = _vector(fixed_times,'fixed_times',positive=True); a = _vector(fixed_accruals,'fixed_accruals',positive=True)
    if t.shape != a.shape or not isinstance(pay_fixed,(bool,np.bool_)):
        raise ValueError('fixed schedule must match; pay_fixed must be bool')
    n = _positive(notional,'notional')
    annuity = float(np.dot(a,_df(discount_curve,t)))
    floating = floating_leg_cashflows(float_starts,float_ends,float_times,float_accruals,
                                     projection_curve=projection_curve,notional=n,spread=spread,fixings=fixings)
    float_pv = float(np.dot(floating['amounts'],_df(discount_curve,floating['times'])))
    fixed_pv = n*_scalar(fixed_rate,'fixed_rate')*annuity
    return {'fixed_pv':fixed_pv,'float_pv':float_pv,'net_pv':(float_pv-fixed_pv)*(1 if pay_fixed else -1),
            'annuity':annuity,'par_rate':float_pv/(n*annuity)}


def bootstrap_projection_curve(instruments, *, discount_curve, interp, extrapolate, bracket, xtol):
    """Calibrate deposit/FRA/IRS projection discount factors under a supplied collateral curve.

    Parameters
    ----------
    instruments : sequence of mappings, increasing unique terminal maturities.
        deposit/fra: kind, start, end, accrual, rate (simple decimal).
        swap: kind='swap', rate, fixed_times, fixed_accruals, float_starts,
        float_ends, float_times, float_accruals; no spread or known fixing.
        These are generic numeric cashflow descriptions, not input-file schemas.
    discount_curve : collateral DF curve; only swap leg PV uses this curve.
    interp, extrapolate : DiscountCurve conventions for the projection curve.
    bracket : continuous node-zero bounds. xtol : positive repricing tolerance.

    Returns
    -------
    DiscountCurve : fitted projection factors, all instrument rate residuals checked.

    Example
    -------
    >>> bootstrap_projection_curve([{'kind':'deposit','start':0,'end':1,'accrual':1,'rate':.04}],discount_curve=lambda t: np.ones_like(t),interp='log_df',extrapolate='flat_zero',bracket=(-.5,1),xtol=1e-12).df(1)
    0.9615384615384616
    """
    rows = list(instruments)
    if not rows:
        raise ValueError('need instruments')
    maturities, guesses = [], []
    for row in rows:
        kind = row['kind']
        if kind in ('deposit','fra'):
            end = _positive(row['end'],'end'); start = _scalar(row['start'],'start')
            a = _positive(row['accrual'],'accrual'); rate = _scalar(row['rate'],'rate')
            if start < 0 or end <= start or 1+rate*a <= 0:
                raise ValueError('invalid deposit/FRA period or simple rate')
            maturity = end
        elif kind == 'swap':
            _float_inputs(row['float_starts'],row['float_ends'],row['float_times'],row['float_accruals'],1,None)
            ft = _vector(row['fixed_times'],'fixed_times',positive=True)
            fa = _vector(row['fixed_accruals'],'fixed_accruals',positive=True)
            if ft.shape != fa.shape:
                raise ValueError('fixed times/accruals must match')
            maturity = max(float(np.max(row['float_ends'])),float(np.max(ft)))
            rate = _scalar(row['rate'],'rate')
        else:
            raise ValueError('instrument kind must be deposit/fra/swap')
        maturities.append(maturity); guesses.append(rate)
    if np.any(np.diff(maturities) <= 0):
        raise ValueError('instrument terminal maturities must increase strictly')
    def residual(curve):
        errors = []
        for row in rows:
            if row['kind'] in ('deposit','fra'):
                rate = (_df(curve,row['start'])/_df(curve,row['end'])-1)/row['accrual']
            else:
                rate = dual_curve_swap(row['fixed_times'],row['fixed_accruals'],row['float_starts'],
                    row['float_ends'],row['float_times'],row['float_accruals'],discount_curve=discount_curve,
                    projection_curve=curve,fixed_rate=row['rate'],notional=1,spread=0,pay_fixed=True,fixings=None)['par_rate']
            errors.append(rate-row['rate'])
        return errors
    return _calibrate(maturities,residual,guesses,interp=interp,extrapolate=extrapolate,bracket=bracket,xtol=xtol)


def mtm_principal_cashflows(reset_times, notionals, *, final_time, initial_exchange, final_exchange):
    """Principal-reset cashflow identity for a varying notional leg.

    Parameters
    ----------
    reset_times, notionals : strictly increasing nonnegative reset times and
        positive notionals fixed at those times.
    final_time : maturity >= last reset. initial_exchange, final_exchange : bool.
        Sign convention: receive initial principal, receive increases at resets,
        return final principal. Multiply all amounts by -1 for the opposite leg.

    Returns
    -------
    dict : times/amounts; reset amounts N_i-N_(i-1). Full exchanges telescope to zero.

    Example
    -------
    >>> mtm_principal_cashflows([0,1],[100,110],final_time=2,initial_exchange=True,final_exchange=True)['amounts'].tolist()
    [100.0, 10.0, -110.0]
    """
    t = _vector(reset_times,'reset_times',nonnegative=True); n = _vector(notionals,'notionals',positive=True)
    last = _scalar(final_time,'final_time')
    if t.shape != n.shape or np.any(np.diff(t) <= 0) or last < t[-1]:
        raise ValueError('reset times must increase/match notionals and precede final time')
    if not all(isinstance(v,(bool,np.bool_)) for v in (initial_exchange,final_exchange)):
        raise ValueError('exchange flags must be bool')
    times, amounts = list(t[1:]),list(np.diff(n))
    if initial_exchange:
        times.insert(0,t[0]); amounts.insert(0,n[0])
    if final_exchange:
        times.append(last); amounts.append(-n[-1])
    return {'times':np.asarray(times,dtype=float),'amounts':np.asarray(amounts,dtype=float)}


def price_cashflows(times, amounts, *, curve, spread):
    """Present value of explicit cashflows with an additive continuous z-spread.

    Parameters
    ----------
    times, amounts : aligned nonnegative year times and signed cashflows.
    curve : DF object/callable. spread : explicit continuous decimal zero spread;
        use 0 for no spread. It multiplies D(t) by exp(-spread*t).

    Returns
    -------
    float : dirty present value, no accrued interest or settled-payment filtering.

    Example
    -------
    >>> price_cashflows([1],[100],curve=lambda t: np.ones_like(t),spread=0)
    100.0
    """
    t,a = _cashflows(times,amounts)
    return float(np.dot(a,_df(curve,t)*np.exp(-_scalar(spread,'spread')*t)))


def curve_cashflow_risk(times, amounts, *, curve, comp, ytm):
    """Curve-weighted cashflow duration, convexity and approximate DV01.

    Parameters
    ----------
    times, amounts : aligned nonnegative times/cashflows. curve : DF curve.
    comp : 'continuous' for exact Fisher-Weil continuous parallel-zero derivatives,
        or positive compounding frequency m for a curve-weighted YTM proxy.
    ytm : None for continuous; explicit decimal YTM for periodic proxy.
        Periodic modified duration = sum(t*PV)/P/(1+y/m); convexity uses
        sum(t*(t+1/m)*PV)/P/(1+y/m)^2. These proxy measures are not derivatives
        of arbitrary curve-node bumps. Use key_rate_risk for actual bumped PV.

    Returns
    -------
    dict : price, duration (PV time mean), modified_duration, convexity,
        dv01=price*modified_duration*1e-4, risk_basis.

    Example
    -------
    >>> curve_cashflow_risk([2],[100],curve=lambda t: np.ones_like(t),comp='continuous',ytm=None)['duration']
    2.0
    """
    t,a = _cashflows(times,amounts); pv=a*_df(curve,t); P=float(np.sum(pv))
    if P <= 0:
        raise ValueError('risk measures need positive net PV')
    D=float(np.dot(t,pv)/P)
    if comp == 'continuous':
        if ytm is not None:
            raise ValueError('continuous curve risk requires ytm=None')
        modified, convexity, basis = D,float(np.dot(t*t,pv)/P),'continuous_zero'
    else:
        if isinstance(comp,bool) or not isinstance(comp,(int,np.integer)) or comp <= 0:
            raise ValueError('periodic comp must be a positive integer')
        den=1+_scalar(ytm,'ytm')/comp
        if den <= 0:
            raise ValueError('1+ytm/comp must be positive')
        modified=D/den; convexity=float(np.dot(t*(t+1/comp),pv)/P/den**2); basis='periodic_ytm_proxy'
    return {'price':P,'duration':D,'modified_duration':modified,'convexity':convexity,
            'dv01':P*modified*1e-4,'risk_basis':basis}


def cashflow_ytm(price, times, amounts, *, comp, bracket, xtol):
    """Solve a dirty-price YTM for arbitrary explicit dated-year cashflows.

    Parameters
    ----------
    price : positive target dirty PV. times, amounts : cashflow arrays.
    comp : explicit discount compounding. bracket : two YTM bounds.
    xtol : positive Brent rate tolerance. Bracket/domain failures raise.

    Returns
    -------
    float : decimal YTM, sum(CF*df_from_rate(y,t,comp)) equals price.

    Example
    -------
    >>> cashflow_ytm(100,[1],[104],comp=1,bracket=(0,.3),xtol=1e-12)
    0.04
    """
    P=_positive(price,'price'); t,a=_cashflows(times,amounts)
    return _root(lambda y:float(np.dot(a,df_from_rate(y,t,comp=comp)))-P,bracket,xtol)


def z_spread(price, times, amounts, *, curve, bracket, xtol):
    """Solve a continuous additive zero spread against a supplied DF curve.

    Parameters
    ----------
    price : positive dirty market price; times/amounts : explicit cashflows.
    curve : base DF curve. bracket : finite increasing spread bounds.
    xtol : positive Brent tolerance. No unbracketed result is returned.

    Returns
    -------
    float : continuous decimal z-spread s with sum(CF*D(t)*exp(-s*t))=price.

    Example
    -------
    >>> z_spread(100,[1],[100],curve=lambda t: np.ones_like(t),bracket=(-.1,.1),xtol=1e-12)
    0.0
    """
    P=_positive(price,'price'); t,a=_cashflows(times,amounts)
    return _root(lambda s:price_cashflows(t,a,curve=curve,spread=s)-P,bracket,xtol)


def _risk_results(price_function, base_object, bumped_objects, *, bump, scheme):
    base=_scalar(price_function(base_object),'base price')
    if base <= 0:
        raise ValueError('duration needs positive base price')
    up=[];down=[]
    for positive,negative in bumped_objects:
        up.append(_scalar(price_function(positive),'shocked price'))
        if scheme == 'central':
            down.append(_scalar(price_function(negative),'shocked price'))
    up=np.asarray(up);down=np.asarray(down) if scheme=='central' else np.full(up.shape,base)
    derivative=(down-up)/(2*bump if scheme=='central' else bump)
    return {'base_price':base,'up_prices':up,'down_prices':down,'duration':derivative/base,
            'dv01':derivative*1e-4,'bump':bump,'scheme':scheme}


def key_rate_risk(price_function, curve, *, key_times, bump, scheme, shape):
    """Spot-zero key-rate risk with explicit bump shape and difference scheme.

    Parameters
    ----------
    price_function : callable taking DiscountCurve, returning positive base PV.
    curve : DiscountCurve. key_times : increasing positive key maturities.
    bump : positive decimal continuous-zero shock (e.g. 1e-4).
    scheme : 'forward' or 'central'. shape : 'node' (each key exactly a curve node)
        or 'triangular' (piecewise-linear hats on key_times, zero outside keys).
        Interpolation/extrapolation of the curve is retained after node shifts.

    Returns
    -------
    dict : base/up/down prices, normalized duration, dv01 (currency per +1bp),
        key_times. Positive dv01 denotes a price decline for a rate increase.
        If bump differs from 1bp, dv01 is the scaled derivative, not raw PV delta.

    Example
    -------
    >>> c=DiscountCurve([1],[.03],input_type='zero',comp='continuous',interp='log_df',extrapolate='flat_zero')
    >>> key_rate_risk(lambda x:100*x.df(1),c,key_times=[1],bump=1e-4,scheme='forward',shape='node')['duration'].shape
    (1,)
    """
    if not isinstance(curve,DiscountCurve):
        raise ValueError('spot key-rate risk requires DiscountCurve')
    keys=_vector(key_times,'key_times',positive=True); b=_positive(bump,'bump')
    if np.any(np.diff(keys) <= 0) or scheme not in ('forward','central') or shape not in ('node','triangular'):
        raise ValueError('invalid keys, scheme or shape')
    hats=[]
    for j,k in enumerate(keys):
        if shape=='node':
            mask=np.isclose(curve.times,k,rtol=0,atol=1e-12)
            if np.sum(mask)!=1:
                raise ValueError('node key must match exactly one original curve time')
            hat=mask.astype(float)
        else:
            if keys.size<2:
                raise ValueError('triangular shape needs at least two keys')
            hat=np.interp(curve.times,keys,np.eye(keys.size)[j],left=0,right=0)
        hats.append(hat)
    objects=[(curve.shifted(b*h),curve.shifted(-b*h) if scheme=='central' else None) for h in hats]
    ans=_risk_results(price_function,curve,objects,bump=b,scheme=scheme)
    ans.update(key_times=keys.copy(),shape=shape,shock_space='continuous_zero')
    return ans


def quote_key_rate_risk(price_function, quotes, *, rebuild, bump, scheme):
    """Quote-space key-rate shocks followed by explicit curve rebuilding.

    Parameters
    ----------
    price_function : callable taking rebuilt curve. quotes : decimal quote vector.
    rebuild : caller's quote-to-curve calibration function; fixes schedules,
        interpolation and discount/projection choices explicitly.
    bump : positive decimal quote shock. scheme : 'forward' or 'central'.

    Returns
    -------
    dict : base/up/down prices, normalized duration and currency DV01 per quote.
        A par-quote shock can alter all zero nodes; this is distinct from key_rate_risk.

    Example
    -------
    >>> build=lambda q: DiscountCurve([1],q,input_type='zero',comp='continuous',interp='log_df',extrapolate='flat_zero')
    >>> quote_key_rate_risk(lambda c:100*c.df(1),[.03],rebuild=build,bump=1e-4,scheme='forward')['dv01'].shape
    (1,)
    """
    q=_vector(quotes,'quotes'); b=_positive(bump,'bump')
    if scheme not in ('forward','central'):
        raise ValueError('scheme must be forward or central')
    objects=[(rebuild(q+b*e),rebuild(q-b*e) if scheme=='central' else None) for e in np.eye(q.size)]
    ans=_risk_results(price_function,rebuild(q.copy()),objects,bump=b,scheme=scheme)
    ans.update(shock_space='quote',quotes=q.copy())
    return ans


def accrued_interest(settlement, previous_coupon, next_coupon, *, coupon_rate, face, frequency, basis, method):
    """Fixed-coupon accrued interest under explicit calendar and stub treatment.

    Parameters
    ----------
    settlement, previous_coupon, next_coupon : dates, previous <= settlement <= next.
    coupon_rate, face : decimal annual coupon and positive face.
    frequency : positive payments/year. basis : dated_year_fraction convention.
    method : 'day_count' (face*coupon*accrued fraction) or 'period_fraction'
        (regular face*coupon/frequency times elapsed/full-period fraction).
        At exact next_coupon, full period accrual is returned; caller must roll
        coupon boundaries for ex-coupon/settlement-date reset conventions.

    Returns
    -------
    float : accrued amount; subtract from dirty price for clean price.

    Example
    -------
    >>> accrued_interest('2024-04-30','2024-01-31','2024-07-31',coupon_rate=.06,face=100,frequency=2,basis='30/360_us',method='day_count')
    1.5
    """
    s,p,n=map(pd.Timestamp,(settlement,previous_coupon,next_coupon))
    if not p <= s <= n or p == n:
        raise ValueError('settlement must lie in a positive coupon period')
    f=_positive(frequency,'frequency'); value=_positive(face,'face')*_scalar(coupon_rate,'coupon_rate')
    elapsed=dated_year_fraction(p,s,basis=basis)
    if method=='day_count':
        return float(value*elapsed)
    if method=='period_fraction':
        return float(value/f*elapsed/dated_year_fraction(p,n,basis=basis))
    raise ValueError('method must be day_count or period_fraction')


def deterministic_call_price_cap(noncall_price, call_times, call_prices, coupon_times, coupon_amounts, *, curve):
    """Deterministic call-date upper bound, not a stochastic callable-bond model.

    Parameters
    ----------
    noncall_price : finite no-call dirty PV. call_times/prices : positive call
        dates and redemption cash amounts; pass only the first call to cap at it.
    coupon_times/amounts : remaining coupons, no final principal; coupons on a
        call date are included. curve : discount-factor curve.

    Returns
    -------
    float : minimum of noncall PV and coupon-through-call + call redemption PV.

    Example
    -------
    >>> deterministic_call_price_cap(110,[1],[100],[1,2],[3,3],curve=lambda t:np.ones_like(t))
    103.0
    """
    ct=_vector(call_times,'call_times',positive=True); cp=_vector(call_prices,'call_prices',positive=True)
    t,a=_cashflows(coupon_times,coupon_amounts)
    if ct.shape!=cp.shape:
        raise ValueError('call times/prices must match')
    values=[_scalar(noncall_price,'noncall_price')]
    for call,redemption in zip(ct,cp):
        keep=t<=call
        values.append(float(np.dot(a[keep],_df(curve,t[keep]))+redemption*_df(curve,call)))
    return min(values)


def fx_forward(spot, base_df, quote_df):
    """Covered-interest-parity forward in quote currency per base currency.

    Parameters
    ----------
    spot : positive quote/base spot. base_df, quote_df : positive discount
        factors for the same maturity in explicitly selected currency curves.

    Returns
    -------
    float or ndarray : F = spot * base_df / quote_df, broadcast over inputs.

    Example
    -------
    >>> fx_forward(1.25,.98,.95)
    1.2894736842105263
    """
    s,b,q=np.broadcast_arrays(np.asarray(spot,float),np.asarray(base_df,float),np.asarray(quote_df,float))
    if not np.isfinite(s).all() or not np.isfinite(b).all() or not np.isfinite(q).all() or np.any(s<=0) or np.any(b<=0) or np.any(q<=0):
        raise ValueError('FX spot and discount factors must be positive finite')
    return _ret(s*b/q)


def fx_implied_basis(spot, forward, base_df, quote_df, time, *, basis_leg, comp):
    """Currency zero-rate basis implied by a supplied forward, with leg sign explicit.

    Parameters
    ----------
    spot, forward : quote/base prices; base_df, quote_df : reference DF factors.
    time : positive year time. basis_leg : 'base' or 'quote', whose zero rate
        is changed holding the other curve fixed. comp : explicit compounding.

    Returns
    -------
    float or ndarray : implied leg zero minus reference leg zero, in decimals.
        This is an FX-implied zero basis, not an IRS/XCCY coupon-spread quote.

    Example
    -------
    >>> fx_implied_basis(1,1,1,1,1,basis_leg='base',comp='continuous')
    0.0
    """
    fx_forward(spot,base_df,quote_df); fx_forward(forward,base_df,quote_df)
    t=np.asarray(time,float)
    if np.any(~np.isfinite(t)) or np.any(t<=0):
        raise ValueError('time must be positive finite')
    s,f,b,q=np.broadcast_arrays(np.asarray(spot,float),np.asarray(forward,float),np.asarray(base_df,float),np.asarray(quote_df,float))
    if basis_leg=='base':
        actual=f/s*q; original=b
    elif basis_leg=='quote':
        actual=s/f*b; original=q
    else:
        raise ValueError('basis_leg must be base or quote')
    return _ret(rate_from_df(actual,t,comp=comp)-rate_from_df(original,t,comp=comp))


_LAM_FORMS = {'time': 'time', 'rate': 'rate', 'decay': 'rate'}


def _bound_pairs(bounds):
    """(n, 2) float bounds; None means no bound on that side (-inf lower, +inf upper)."""
    try:
        return np.array([[-np.inf if lo is None else lo, np.inf if hi is None else hi] for lo, hi in bounds], float)
    except (TypeError, ValueError):
        raise ValueError('bounds must have one increasing pair per free parameter') from None


def _ns_fit_jacobian(t, beta, lam, *, form, model, free_lams):
    """d(ns_yield)/d[betas, free lambdas], (len(t), k): x = tau/lam with dx/dlam = -x/lam ('time'), or x = lam*tau
    with dx/dlam = tau ('rate')."""
    lam=np.asarray(lam,dtype=float)
    x=t[:,None]/lam if form=='time' else t[:,None]*lam
    ex=np.exp(-x)
    L=-np.expm1(-x)/x
    M=L-ex
    # Cancellation-free derivative near x=0, d[(1-exp(-x))/x]/dx.
    dL=np.where(np.abs(x)<1e-4,-.5+x/3-x*x/8+x*x*x/30,
                (np.expm1(-x)+x*ex)/(x*x))
    dM=dL+ex
    columns=[np.ones(t.size),L[:,0],M[:,0]]
    if model=='svensson':
        columns.append(M[:,1])
    if free_lams:
        dx=-x/lam if form=='time' else np.broadcast_to(t[:,None],x.shape)
        columns.append((beta[1]*dL[:,0]+beta[2]*dM[:,0])*dx[:,0])
        if model=='svensson':
            columns.append(beta[3]*dM[:,1]*dx[:,1])
    return np.column_stack(columns)


def fit_parametric_curve(tau, yields, *, model, lam_form, method, x0, bounds, options, fixed_lams=None, multistarts=(),
                         gradient='analytic'):
    """NS/Svensson least-squares fit with auditable optimizer and primary start.

    lam_form 'time' is x = tau/lam (Nelson-Siegel 1987) and 'rate' is x = lam*tau
    (Diebold-Li; 'decay' is accepted as an alias of 'rate'). gradient='2-point' gives
    the optimizer scipy's own finite differences instead of the analytic Jacobian, which
    reproduces a TASK that prescribes plain scipy.optimize.minimize(method='L-BFGS-B',
    options=...) or least_squares from given starts.

    Parameters
    ----------
    tau, yields : aligned positive maturities and finite yields (same units as betas).
    model : 'ns' (3 betas,1 lambda) or 'svensson' (4 betas,2 lambdas).
    lam_form : 'time' or 'rate' (ns_yield convention; 'decay' = 'rate'). method : 'L-BFGS-B'
        on unscaled sum of squared errors or 'least_squares' on yield residuals.
    x0 : explicit primary vector [betas...,lambdas...] or only betas when fixed_lams.
    bounds : explicit (lower,upper) per free parameter, None for no bound on that side;
        lambda lower must be >0.
    options : scipy optimizer options, e.g. ftol/gtol/maxiter or ftol/xtol/gtol.
    fixed_lams : optional positive fixed lambda vector. multistarts : additional
        vectors; primary is always evaluated first, no hidden start generation.
    gradient : 'analytic' (default, exact Jacobian) or '2-point' (L-BFGS-B jac=None,
        least_squares jac='2-point': scipy's finite differences).

    Returns
    -------
    dict : params, betas, lams, fitted, rmse, method, success, message, nfev,
        starts (all initial vectors, convergence flags, objectives). Prefer the
        lowest-SSE successful start; if none converge, success remains False.
        Parameter identification can be weak even when curve fit is accurate.

    Example
    -------
    >>> fit_parametric_curve([1,2,3],[.03]*3,model='ns',lam_form='time',method='least_squares',x0=[.02,0,0],bounds=[(-1,1)]*3,options={},fixed_lams=[1])['success']
    True
    """
    t=_vector(tau,'tau',positive=True); y=_vector(yields,'yields')
    if t.shape!=y.shape or model not in ('ns','svensson') or method not in ('L-BFGS-B','least_squares'):
        raise ValueError('invalid model/method or unmatched observations')
    form=_LAM_FORMS.get(str(lam_form).strip().lower())
    if form is None:
        raise ValueError("lam_form must be 'time' (x = tau/lam, NS 1987) or 'rate' (x = lam*tau, Diebold-Li; "
                         "alias 'decay')")
    if gradient not in ('analytic','2-point'):
        raise ValueError("gradient must be 'analytic' or '2-point'")
    nb,nl=(3,1) if model=='ns' else (4,2)
    fixed=None if fixed_lams is None else _vector(fixed_lams,'fixed_lams',positive=True)
    if fixed is not None and len(fixed)!=nl:
        raise ValueError('wrong fixed lambda count')
    nfree=nb+(nl if fixed is None else 0)
    limits=_bound_pairs(bounds)
    if limits.shape!=(nfree,2) or np.any(np.isnan(limits)) or np.any(limits[:,0]>=limits[:,1]):
        raise ValueError('bounds must have one increasing pair per free parameter')
    if fixed is None and np.any(limits[nb:,0]<=0):
        raise ValueError('free lambda lower bounds must be positive')
    def split(p):
        return p[:nb],(p[nb:] if fixed is None else fixed)
    def residual(p):
        beta,lam=split(p)
        return np.asarray(ns_yield(t,beta,lam,lam_form=form))-y
    def jacobian(p):
        beta,lam=split(p)
        return _ns_fit_jacobian(t,beta,lam,form=form,model=model,free_lams=fixed is None)
    analytic=gradient=='analytic'
    results=[];receipts=[]
    for initial in [x0,*list(multistarts)]:
        p=_vector(initial,'initial')
        if p.shape!=(nfree,) or np.any(p<limits[:,0]) or np.any(p>limits[:,1]):
            raise ValueError('initial vector outside bounds or wrong size')
        if method=='least_squares':
            res=optimize.least_squares(residual,p,jac=jacobian if analytic else '2-point',
                                       bounds=(limits[:,0],limits[:,1]),**dict(options))
        else:
            res=optimize.minimize(lambda v:float(np.dot(residual(v),residual(v))),p,
                                  jac=(lambda v:2*jacobian(v).T@residual(v)) if analytic else None,
                                  method='L-BFGS-B',bounds=limits,options=dict(options))
        sse=float(np.dot(residual(res.x),residual(res.x)))
        success=bool(res.success and np.isfinite(sse))
        results.append((res,sse,success))
        receipts.append({'initial':p.tolist(),'success':success,'sse':sse,'message':str(res.message),'nfev':int(res.nfev)})
    successful=[j for j,(_,_,ok) in enumerate(results) if ok]
    index=min(successful or list(range(len(results))),key=lambda j:results[j][1])
    res,sse,success=results[index];beta,lam=split(res.x)
    return {'params':res.x.copy(),'betas':beta.copy(),'lams':lam.copy(),'fitted':y+residual(res.x),
            'rmse':math.sqrt(sse/y.size),'method':method,'success':success,'message':str(res.message),
            'nfev':int(res.nfev),'starts':receipts,'chosen_start':index}
