"""qfkit: standard textbook quant-finance building blocks (numpy / scipy / pandas / statsmodels).

General components only: option pricing (bsm), exotic closed forms (exotics), trees,
finite differences, American approximations and Monte Carlo (numerics), stochastic and
local volatility (volsurf), rate curves and bonds (curves), short-rate models and swaps
(rates), credit (credit), regression econometrics (econ), volatility estimators and
time-series tests (riskstats), short-rate and jump processes (processes), market risk
(risk), portfolio construction (portfolio), copulas (copulas) and supplements: the BAW
exercise boundary, forward-start and cliquet options, mean-reverting jump moments
(extras); delta-hedge ledgers and escrowed-dividend pricing (hedging); a finite-difference
PDE engine with American exercise and Richardson extrapolation (pde); realized measures and
microstructure-noise corrections (realized); Ornstein-Uhlenbeck fitting and simulation (ou);
backtest ledgers (ledger); event studies (events); VaR/ES forecasting and backtests (varbt); 0.5.0, ported
from the team's qflib: Gaussian hidden Markov models (hmm); EMA and cross-sectional signal tools (signals);
absorption ratio and Marchenko-Pastur edge (spectral); CreditMetrics migrations, latent-factor defaults, the
Vasicek ASRF model and Euler ES contributions (creditpf); copula fit and selection (copfit); Panjer/FFT compound
losses (aggloss); SVI smiles and Gatheral local volatility (svi). Version 0.6.0 adds
barrier/Monte Carlo diagnostics, explicit dated curve and risk conventions, factor
constraints, execution allocation, actual arch estimators and numerical validation tools.
Version 0.6.1 adds explicit date/numeric-row and lagged-sample adapters.
Version 0.6.2 adds labelled panels with explicit axis order and missing policies.
Every market convention (compounding,
day count, units, quantile rule, lag rule, ...) is an explicit argument; the few
defaults are the textbook ones and are documented. No I/O, no network, deterministic.

Every public name is importable from the package (``qfkit.bsm_price``) or from
its module (``qfkit.bsm.bsm_price``). ``CATALOG`` is a one-line-per-function
index; ``help(qfkit.<name>)`` gives parameters, units, formula and an example.
"""

from . import (bsm, copulas, credit, curves, econ, exotics, extras, numerics, portfolio, processes, rates, risk,
               riskstats, volsurf)
from . import events, hedging, ledger, ou, pde, realized, varbt  # 0.4.0 modules
from . import aggloss, copfit, creditpf, hmm, signals, spectral, svi  # 0.5.0 modules
from . import attribution, barrier_tools, mc_tools, pricing_ext, tail_tools, rate_tree, iv_tools, timeseries_tools, execution, arch_tools, rates_ext, factors, risk_ext  # 0.6.0 additions
from . import input_tools, streaming, calendar_input
from .bsm import *  # noqa: F401,F403
from .curves import *  # noqa: F401,F403
from .econ import *  # noqa: F401,F403
from .processes import *  # noqa: F401,F403
from .risk import *  # noqa: F401,F403
from .exotics import *  # noqa: F401,F403
from .numerics import *  # noqa: F401,F403
from .volsurf import *  # noqa: F401,F403
from .rates import *  # noqa: F401,F403
from .credit import *  # noqa: F401,F403
from .riskstats import *  # noqa: F401,F403
from .portfolio import *  # noqa: F401,F403
from .copulas import *  # noqa: F401,F403
from .extras import *  # noqa: F401,F403
from .hedging import *  # noqa: F401,F403
from .pde import *  # noqa: F401,F403
from .realized import *  # noqa: F401,F403
from .ou import *  # noqa: F401,F403
from .ledger import *  # noqa: F401,F403
from .events import *  # noqa: F401,F403
from .varbt import *  # noqa: F401,F403
from .hmm import *  # noqa: F401,F403
from .signals import *  # noqa: F401,F403
from .spectral import *  # noqa: F401,F403
from .creditpf import *  # noqa: F401,F403
from .copfit import *  # noqa: F401,F403
from .aggloss import *  # noqa: F401,F403
from .svi import *  # noqa: F401,F403
from .barrier_tools import *  # noqa: F401,F403
from .mc_tools import *  # noqa: F401,F403
from .attribution import *  # noqa: F401,F403
from .pricing_ext import *  # noqa: F401,F403
from .tail_tools import *  # noqa: F401,F403
from .rate_tree import *  # noqa: F401,F403
from .iv_tools import *  # noqa: F401,F403
from .timeseries_tools import *  # noqa: F401,F403
from .execution import *  # noqa: F401,F403
from .arch_tools import *  # noqa: F401,F403
from .rates_ext import *  # noqa: F401,F403
from .factors import *  # noqa: F401,F403
from .risk_ext import *  # noqa: F401,F403
from .input_tools import *  # noqa: F401,F403
from .streaming import *  # noqa: F401,F403
from .calendar_input import *  # noqa: F401,F403

__version__ = "0.6.2"

# The first five are qfkit 0.1.0, the next seven 0.2.0, the next two 0.3.0, then 0.4.0's and 0.5.0's (CATALOG keeps
# that order; 0.4.0 also appends asian_curran_price to extras).
MODULES = (bsm, curves, econ, processes, risk, exotics, numerics, volsurf, rates, credit, riskstats, portfolio,
           copulas, extras,
           hedging, pde, realized, ou, ledger, events, varbt,
           hmm, signals, spectral, creditpf, copfit, aggloss, svi,
           barrier_tools, mc_tools, attribution, pricing_ext, tail_tools, rate_tree, iv_tools, timeseries_tools, execution, arch_tools, rates_ext, factors, risk_ext, input_tools, streaming, calendar_input)

__all__ = [_name for _mod in MODULES for _name in _mod.__all__]

CATALOG = """\
qfkit: textbook quant blocks; help(qfkit.<name>) = units, formula, example.
After * keyword-only; shown without default = REQUIRED; unlisted args optional.

[bsm] T years; r,q,sigma continuous annual decimals; kind='call'|'put'
bsm_d1d2(S,K,T,r,sigma,*,q=0) - BSM d1, d2
bsm_price(S,K,T,r,sigma,*,kind,q=0) - European price
bsm_greeks(S,K,T,r,sigma,*,kind,q=0,vega_per='unit',theta_per='year',rho_per='unit') - Greeks dict (theta_per 'calendar_day'|'trading_day'|N; vega/rho 'percent')
bsm_pde_residual(S,sigma,r,price,delta,gamma,theta,*,q=0) - BSM PDE residual (annual theta)
parity_residual(call,put,S,K,T,r,*,q=0) - C-P-(Se^-qT - Ke^-rT)
bsm_bounds(S,K,T,r,*,kind,q=0) - no-arbitrage (lower, upper)
implied_vol(price,S,K,T,r,*,kind,q=0) - BSM implied vol (nan if no root)
black76_price(F,K,T,sigma,*,df,kind) - Black-76 on forward/futures, caplets (df to payment)
black76_implied_vol(price,F,K,T,*,df,kind) - Black-76 implied vol

[curves] decimals, years; comp='continuous'|'simple'|int m (or 'annual','semiannual','quarterly','monthly')
year_fraction(start,end,*,basis) - day count ('act/365f'|'act/360'|'act/act_isda'|'30/360_us'|'30e/360')
df_from_rate(rate,t,*,comp) - discount factor
rate_from_df(df,t,*,comp) - zero rate
convert_rate(rate,*,from_comp,to_comp,t=None) - equal-DF conversion (t for 'simple')
forward_rate(t1,t2,df1,df2,*,comp) - forward rate on [t1,t2]
ZeroCurve(t,zero,*,comp,interp,extrapolate='flat') - curve .df .zero .forward .par_yield (interp 'linear_zero'|'log_df')
bootstrap_zero_curve(maturities,coupon_rates,*,freq,comp,interp,prices=None) - exact par/price bootstrap -> ZeroCurve
par_yield(curve,T,*,freq) - par coupon rate
bond_cashflows(T,coupon_rate,*,freq,face=100) - coupon times back from T, amounts
bond_price(T,coupon_rate,ytm,*,freq,comp,face=100) - full price from yield (market: comp=freq)
bond_price_curve(T,coupon_rate,curve,*,freq,face=100) - full price on a curve
bond_ytm(price,T,coupon_rate,*,freq,comp,face=100) - yield from full price
bond_risk(T,coupon_rate,ytm,*,freq,comp,face=100) - price, Macaulay/modified duration, convexity (yr^2), DV01
ns_yield(tau,betas,lams,*,lam_form) - NS/Svensson yield (lam_form 'rate' lam*tau | 'time' tau/lam)
fit_ns(tau,y,*,model,lam_form,lam=None,lam_bounds=None,weights=None) - 'ns'/'svensson' fit; fix lam or give lam_bounds

[econ] rows time-ordered; k includes intercept; add_const -> 'const' first
ols(y,X,*,add_const,cov='classical',lags=None,small_sample=False,dist='t') - OLS (cov 'HC0'|'HC1'|'HAC'; dist 'normal')
cov_ols(X,resid,*,kind,lags=None,small_sample=False) - coef covariance (HAC Bartlett 1-j/(L+1))
newey_west_lags(n,*,rule,rounding='floor') - HAC lags ('nw1994' 4(n/100)^(2/9) | 'sw' 0.75n^(1/3))
t_test(params,cov,*,df=None,null=0) - se, t, two-sided p (df None -> normal)
r_squared(y,fitted,*,k,centered=True) - R^2, adj R^2
durbin_watson(resid) - Durbin-Watson
grs_test(excess_returns,factors,*,variant='mle') - GRS joint F-test of zero alphas -> dict stat, pvalue (floats), df1, df2
vif(X) - variance inflation factors (X without constant)
rolling_ols(y,X,window,*,add_const,min_nobs=None) - rolling OLS betas

[processes] dt = years per step
cir_logpdf(r_next,r_prev,kappa,theta,sigma,*,dt,density_of='rate') - exact CIR transition log-density
cir_moments(r_prev,kappa,theta,sigma,*,dt) - CIR conditional mean, var
cir_bond_price(r,tau,kappa,theta,sigma,*,lam=0) - CIR zero-coupon price
cir_euler_start(r,*,dt) - CIR MLE start values
feller_ratio(kappa,theta,sigma) - 2 kappa theta/sigma^2
ou_moments(x_prev,kappa,theta,sigma,*,dt) - OU conditional mean, var
ou_fit_ar1(x,*,dt) - exact OU/Vasicek MLE via AR(1)
vasicek_logpdf(r_next,r_prev,kappa,theta,sigma,*,dt) - Vasicek transition log-density
vasicek_bond_price(r,tau,kappa,theta,sigma) - Vasicek zero-coupon price (risk-neutral theta)
poisson_truncation(mean,*,tol=1e-12) - jump-count cutoff N
merton_logpdf(x,mu,sigma,lam,mu_j,sigma_j,*,dt,drift='price') - Merton log-return log-density (drift 'log')
merton_price(S,K,T,r,sigma,lam,mu_j,sigma_j,*,kind,q=0) - Merton jump-diffusion option price
multistart_minimize(fun,bounds,*,n_starts=8,x0=None,seed=0) - best of seeded local fits

[risk] LOSSES (pass -returns); alpha = confidence (0.99)
var_es_historical(losses,alpha,*,method,es_rule) - empirical VaR/ES (numpy quantile method; es_rule 'ge'|'gt'|'k_worst')
var_es_normal(alpha,*,sigma,mu=0) - normal VaR/ES
var_es_t(alpha,nu,*,mu=0,scale=None,sd=None) - Student-t VaR/ES (give scale or sd)
var_cornish_fisher(alpha,*,sigma,skew,excess_kurt,mu=0) - Cornish-Fisher VaR (loss skew)
kupiec_pof(n_exceed,n_obs,p) - Kupiec POF test (p=1-alpha)
christoffersen_counts(hits) - transition counts n00..n11
christoffersen_ind(hits=None,*,counts=None) - independence LR test
christoffersen_cc(hits,p,*,uc_sample='full') - conditional coverage LR test
gpd_fit(excesses) - GPD MLE of L[L>u]-u
pot_var_es(alpha,*,u,xi,beta,n,n_u) - POT VaR/ES
hill_estimator(losses,k,*,ref) - Hill xi (ref 'k+1'|'k')
mean_excess(losses,u) - mean excess

[exotics] as [bsm]; continuous monitoring; s_ext = the payoff's running min/max so far (S if new)
barrier_price(S,K,H,T,r,sigma,*,kind,barrier,rebate=0,rebate_timing=None,q=0,monitor_dt=None) - barrier 'down-in'|'down-out'|'up-in'|'up-out'; rebate_timing 'hit'|'expiry'; monitor_dt: discrete (BGK)
digital_price(S,K,T,r,sigma,*,kind,payoff,cash=1,q=0) - payoff 'cash'|'asset'
lookback_floating_price(S,s_ext,T,r,sigma,*,kind,q=0) - call S_T-min, put max-S_T
lookback_fixed_price(S,K,s_ext,T,r,sigma,*,kind,q=0) - call max-K, put K-min
asian_geometric_price(S,K,T,r,sigma,*,kind,fixings,q=0) - fixings 'continuous'|times
asian_tw_price(S,K,T,r,sigma,*,kind,q=0,avg_start=0,realized_avg=None,past_weight=0) - Turnbull-Wakeman, continuous average
asian_levy_price(S,K,T,r,sigma,*,kind,fixings,q=0) - Levy; seasoned as tw
margrabe_price(S1,S2,T,sigma1,sigma2,rho,*,q1=0,q2=0) - max(S1_T-S2_T,0)
kirk_spread_price(F1,F2,K,T,sigma1,sigma2,rho,*,df,kind) - call max(F1-F2-K,0)
geske_compound_price(S,K1,K2,T1,T2,r,sigma,*,kind,underlying,q=0) - kind (K1,T1) on underlying (K2,T2)
chooser_price(S,K,t_choose,T,r,sigma,*,q=0) - simple chooser

[numerics] as [bsm]; exercise 'european'|'american'
crr_price(S,K,T,r,sigma,*,kind,exercise,steps,q=0,greeks=False) - CRR (greeks dict, theta/yr)
fd_cn_price(S,K,T,r,sigma,*,kind,exercise,s_max,n_space,n_time,q=0) - Crank-Nicolson
baw_price(S,K,T,r,sigma,*,kind,q=0) - Barone-Adesi-Whaley American
bs2002_price(S,K,T,r,sigma,*,kind,q=0) - Bjerksund-Stensland 2002 American
gbm_paths(S,T,mu,sigma,*,n_paths,n_steps,seed) - (n_paths,n_steps+1), col 0 = S
gbm_mc_price(S,T,r,sigma,*,payoff,n_paths,n_steps,seed,estimator,q=0) - payoff(paths) paid at T; 'plain'|'antithetic'|'control_variate' -> dict price,se,ci95

[volsurf] as [bsm]; Heston v0,theta are variances
heston_price(S,K,T,r,v0,kappa,theta,sigma_v,rho,*,kind,q=0) - European
sabr_implied_vol(F,K,T,alpha,beta,rho,nu,*,vol_type) - Hagan; 'lognormal'|'normal'
iv_brenner_subrahmanyam(price,S,T,*,q=0) - ATM-forward approx
iv_corrado_miller(price,S,K,T,r,*,kind,on_negative,q=0) - on_negative 'nan'|'zero'
dupire_local_vol(strikes,maturities,surface,*,S,r,surface_type,q=0) - local vols; surface (T,K) of call prices|ivs: 'price'|'iv'

[rates] curve=ZeroCurve|t->P(0,t); sigma normal short-rate vol; theta risk-neutral
vasicek_bond_option(r,T_opt,T_bond,K,kappa,theta,sigma,*,kind,face=1) - ZCB option
hw_bond_price(r_t,t,T,a,sigma,*,curve) - Hull-White P(t,T)
hw_bond_option(K,T_opt,T_bond,a,sigma,*,curve,kind,face=1) - Hull-White ZCB option
hw_caplet_price(K,t_reset,t_pay,a,sigma,*,curve,kind) - 'caplet'|'floorlet'
hw_swaption_price(K,T_opt,pay_times,a,sigma,*,curve,kind) - 'payer'|'receiver'
swap_annuity(start,pay_times,*,curve,accruals=None) - sum tau_i P(0,t_i)
par_swap_rate(start,pay_times,*,curve,accruals=None) - (P(start)-P(t_n))/annuity

[credit] decimals; hazard knots = right ends; default_timing 'mid_period'|'continuous'
merton_structural(V,D,T,r,sigma_V,*,mu=None) - dict equity,debt,pd_rn,expected_loss,recovery,spread
merton_implied_asset(E,sigma_E,D,T,r) - dict V,sigma_V + merton keys
hazard_from_spread(spread,*,recovery) - s/(1-R)
survival_probability(t,hazard_times,hazards) - Q(t)
cds_value(spread,hazard_times,hazards,*,pay_times,recovery,curve,default_timing,accrual_on_default) - dict par_spread,value (buyer),risky_annuity
bootstrap_hazard(maturities,spreads,*,pay_freq,recovery,curve,default_timing,accrual_on_default) - dict hazards
migration_n_period(P,n) - P^n
migration_generator(P,*,method) - 'log'|'diagonal_adjust'|'weighted_adjust'|'jlt'
migration_from_generator(Q,t) - expm(Qt)

[riskstats] time-ordered returns per period; variances in squared return units
garch11_fit(returns,*,mean,dist,cov_type) - 'constant'|'zero'; 'normal'|'t'; 'robust'|'classic' -> dict params,se,cond_vol,var_next
garch11_forecast(omega,alpha,beta,*,var_next,horizon) - dict variance,cumulative
ewma_vol(returns,*,lam,init,timing) - init 'first_sq'|'sample_var'|variance; 'forecast'|'filter'
ewma_cov(returns,*,lam,init,timing) - (T,N,N)
range_vol(open,high,low,close,*,estimator,periods_per_year,window=None) - p.a. 'parkinson'|'garman_klass'|'rogers_satchell'|'yang_zhang'
realized_variance(returns) - sum r^2 (days x intraday -> per day)
bipower_variation(returns,*,scaling) - 'none'|'n_over_n_minus_1'
engle_granger(y,x,*,trend,maxlag,autolag) - as statsmodels coint -> dict stat,pvalue,crit
ar1_half_life(x,*,method,dt) - in dt units; 'ar1'|'diff'
fama_macbeth(y,X,*,add_const,se,lags=None) - 'fm'|'newey_west' -> dict lambda,se,t,pvalue

[portfolio] mu,cov,rf per period; weights sum 1 unless noted
min_variance_weights(cov,*,long_only) - GMV
tangency_weights(mu,cov,*,rf,long_only) - max Sharpe
target_return_weights(mu,cov,target,*,long_only) - frontier
risk_parity_weights(cov,*,budgets=None) - ERC
risk_contributions(w,cov,*,kind) - 'marginal'|'component'|'percent'
bl_implied_returns(cov,w_mkt,*,delta) - delta cov w_mkt
black_litterman(cov,pi,P,Q,*,tau,omega,delta=None) - omega matrix|'he_litterman' -> dict mean,cov_pred=cov+M,weights=solve(delta*cov_pred,mean) (He-Litterman, not delta*cov)
kelly_bet(p,b,*,a=1.0) - p/a-(1-p)/b
kelly_weights(mu,cov,*,rf) - cov^-1(mu-rf), not normalised (sum>1 = leverage)
sharpe_ratio(returns,*,rf,periods_per_year,ddof) - annualised
sortino_ratio(returns,*,target,periods_per_year,downside) - 'full'|'below_target'
max_drawdown(x,*,input_type) - 'prices'|'returns'|'log_returns' -> dict max_drawdown
calmar_ratio(returns,*,periods_per_year,ann_return) - 'cagr'|'arithmetic'

[copulas] u (n,d) in (0,1); family 'gaussian'|'t'|'clayton'|'gumbel'|'frank'; theta=rho|corr matrix for gaussian/t; 't' needs nu=
copula_tau(theta,*,family) - Kendall tau
copula_theta(tau,*,family) - theta from Kendall tau
copula_sample(n,theta,*,family,seed,dim=2,nu=None) - (n,dim); Archimedean by Marshall-Olkin; nu required for family='t', otherwise omit or None
copula_logpdf(u,theta,*,family,nu=None) - log density per row (Archimedean d=2); nu required for family='t', otherwise omit or None
copula_cdf(u,theta,*,family,nu=None) - C(u) (gaussian/t d=2); nu required for family='t', otherwise omit or None
copula_tail_dependence(theta,*,family,nu=None) - dict lower,upper: asymptotic limits, not finite-threshold empirical frequencies; nu required for family='t', otherwise omit or None

[extras] as [bsm]; times in years from today
baw_critical_price(K,T,r,sigma,*,kind,q=0) - BAW exercise boundary S* (call S>=S*, put S<=S*)
forward_start_price(S,t_start,T,r,sigma,*,kind,alpha,q=0) - strike alpha*S(t_start)
cliquet_price(S,reset_times,r,sigma,*,kind,payoff,payment,alpha=1,q=0) - forward-start sum over (t_i-1,t_i], t_0=0; call pays 'price' (S_i-alpha*S_i-1)+|'return' (S_i/S_i-1-alpha)+; payment 'period_end'|'maturity' -> dict price,legs
ou_jump_moments(x_prev,kappa,theta,sigma,lam,mu_j,sigma_j,*,dt,compensated) - dX=kappa(theta-X)dt+sigma dW+J dN, J~N(mu_j,sigma_j^2), compensated adds -lam*mu_j dt -> dict mean,var,skew,ex_kurt,mean_exp=E[e^X]
asian_curran_price(S,K,T,r,sigma,*,kind,fixings,q=0) - Curran arithmetic Asian (fixings = times)

[hedging] steps t=0..N; position<0 short; tc bps
delta_hedge_pnl(S_path,*,option_value,hedge_ratio,payoff,position,rebalance_every,dt,r,step_order,interest_comp,dividends,dividend_timing,tc_buy_bps,tc_sell_bps,rebalance_at_expiry,t_remaining) - step_order e.g. ('interest','dividend','rebalance') -> dict pnl,total_tc,ledger
bsm_escrowed_dividend(S,K,t_rem,r,sigma,*,kind,dividends) - BSM on S-PV(divs [(tau,D)]) -> dict price,delta

[pde] S grid i*s_max/n_space; None if unused
fd_solve(S,K,T,r,sigma,*,kind,exercise,scheme,s_max,n_space,n_time,boundary,american_method,tol,omega,exercise_tol,return_grid,q=0,dividends=None,dividend_interp=None) - 'cn'|'implicit'|'explicit'; 'psor'|'brennan_schwartz'|'projection' -> dict price,delta,gamma,theta,exercise_boundary; dividend_interp 'linear'|'cubic' required when a dividend is applied
richardson(fine,coarse,*,order) - fine+(fine-coarse)/(ratio^order-1)

[realized] log returns per day; n=returns
subsample_grid(timestamps,prices,*,every_minutes,anchor,rule) - anchor 'HH:MM'; 'on_grid'|'previous_tick'
realized_measures(day_prices,*,grids,anchor,rule,noise_estimator,correction,omega_source,bv_scaling) - 'additive'|'tsrv'|'none'; 'grand_mean'|'per_day' -> dict table,omega2_grand
noise_variance(returns,*,method) - 'bandi_russell' RV/(2n)|'autocov'
noise_corrected_rv(rv,n,omega2,*,method) - 'additive' RV-2n*omega2|'tsrv'

[ou] dX=kappa(theta-X)dt+sigma dW
ou_fit(x,*,dt,method,sigma_method,resid_ddof,pair_mask=None) - 'euler_diff'|'exact_ar1'; sigma 'euler'|'exact' -> dict kappa,theta,sigma,half_life_periods
ou_simulate(x0,kappa,theta,sigma,*,dt,n_steps,n_paths,rng,scheme,record,jumps=None,jump_scheme=None) - 'exact'|'euler'; 'before_advance'|'after_advance'; jumps {lam,mu_j,sigma_j}
detect_jumps(resid,*,k_sigma,std_ddof,reestimate_sigma) - |resid|>k*std -> dict mask,n_jumps,mu_j,sigma_j

[ledger] signal/weight set at close of row t
signal_trade_ledger(dates,mark_price,exec_price,*,entry,exit,execution,sizing,fee_fixed,fee_pct,initial_cash,forced_close,block_while_pending,stop=None) - 'next_open'|'next_close'|'same_close' -> dict trades,equity,returns
weights_ledger(asset_returns,target_weights,*,lag,rebalance,drift,cost_bps,cost_base,cost_timing) - row t earns r[t+lag] -> dict returns,weights_bop,turnover,costs

[events] windows (a,b) trading-day offsets, inclusive
event_study(returns,market,events,*,estimation_window,event_window,model,min_est_obs,require_full_event_window,event_day_rule) - per event -> dict params,ar,car,caar
event_study_tests(res,*,tests,patell_correction,car_standardization,kp_formula,kp_alignment,kp_min_overlap,rank_method,rank_scaling,rank_aggregation) - Patell,BMP,Kolari-Pynnonen,Corrado -> dict stats,by_day

[varbt] LOSSES; alpha=confidence
var_es_forecast(losses,alpha,*,method,quantile_method,es_rule,mu_sign) - 'historical'|'normal'|'t'|'cornish_fisher'|'evt_pot'|'ewma_normal'|'garch_{normal,t,evt,fhs}'; es 'ge'|'gt' -> dict var,es
filtered_var_es(returns,alpha,*,garch_mean,garch_dist,input_scale,tail,threshold_quantile,quantile_method,es_rule,vol_at) - GARCH 'evt'|'fhs' -> dict var,es,sigma
rolling_var_backtest(losses,*,window,alpha,method,refit_every,horizon,sqrt_time,quantile_method,es_rule,mu_sign,hit_rule) - VaR_t from losses[t-window:t] -> dict var,hits,kupiec,christoffersen_cc

[hmm] var = variances, not sd
gaussian_hmm_fit(x,*,mu0,var0,trans0,pi0,tol,max_iter,converge,state_order) - Baum-Welch; converge 'mu'|'loglik'; state_order 'as_given'|'mean'|'variance' -> dict mu,sd,gamma,n_iter
hmm_posterior(x,*,mu,var,trans,pi) - forward-backward, Viterbi -> dict smoothed,viterbi,loglik

[signals] rows=dates, cols=assets, NaN skipped
ema(x,*,span,init) - alpha=2/(span+1); init 'first'|'sma'|'adjusted'
cs_zscore(x,*,ddof,zero_std) - z-score; zero_std 'zero'|'nan'
cs_winsorize(x,*,lower,upper,method) - clip at row quantiles
quantile_buckets(x,*,n_buckets,rule) - 1..n, 1=lowest; 'ordinal'|'quantile'
information_coefficient(signal,forward_returns,*,method,min_obs,ddof) - 'spearman'|'pearson' -> dict ic,mean,std,ir

[spectral] x eigenvalues or cov/corr matrix
absorption_ratio(x,*,k) - share of the k largest eigenvalues
marchenko_pastur_edge(n_obs,n_vars,*,sigma2) - dict lambda_min,lambda_max

[creditpf] P cols best..default; rng Generator|RandomState|int
migration_thresholds(P,*,order) - CreditMetrics; 'best_to_worst'|'worst_to_best'
simulate_migrations(P,start,*,rho,n_paths,rng,draws) - 'blocks'|'per_path' -> dict ratings,default
latent_factor_defaults(pd,*,rho,n_paths,rng,nu) - Gaussian (nu None)|t copula; factor_corr,factor
vasicek_loss_cdf(x,pd,*,lgd,rho) - large-pool loss CDF
vasicek_loss_quantile(alpha,pd,*,lgd,rho,weight=None) - ASRF LGD N(h); weight= sums obligors
es_contributions(X,alpha,*,weights,var_rule,es_rule) - Euler ES split of L=X@weights -> dict var,es,contrib

[copfit] u (n,2) pseudo-obs; family as [copulas]
pseudo_observations(x,*,ties) - rank/(n+1) per column
copula_fit(u,*,family,method,nu=None) - 'mle'|'tau' -> dict theta,nu,loglik,aic,bic
information_criteria(loglik,*,n,n_params) - dict aic,bic
empirical_tail_dependence(u,*,q) - dict upper,lower

[aggloss] grid 0,h,..; freq ('poisson',lam)|('negative_binomial',r,beta)|('binomial',n,p)
discretise_severity(cdf,*,h,n_points,method) - 'rounding'|'lower'|'upper' -> dict pmf,grid
panjer_recursion(f,*,freq) - Panjer aggregate pmf
compound_fft(f,*,freq,pad) - FFT aggregate pmf, buffer pad*len(f)
pmf_var_es(pmf,grid,alpha,*,es_rule) - VaR min{x:F>=alpha}; 'ge'|'gt' -> dict var,es

[svi] k=ln(K/F_T); w=total variance iv^2 T
svi_total_variance(k,a,b,rho,m,sigma,*,deriv=0) - raw SVI w; deriv 1|2: w', w''
svi_calibrate(k,w,*,lower,upper,weights=None) - bounds (a,b,rho,m,sigma) -> dict a,b,rho,m,sigma,rmse
gatheral_local_vol(k,w,w_k,w_kk,w_T) - Dupire local vol from total variance

[barrier_tools] Brownian log-space; time years; explicit direction and surface conventions
brownian_hit_cdf(barrier,t,mu,sigma,*,direction) - first-passage CDF; direction 'up'|'down', barrier distance >=0
brownian_hit_pdf(barrier,t,mu,sigma,*,direction) - continuous first-passage density per year
brownian_joint_cdf(x,maximum,t,mu,sigma) - P(X_t<=x, running max<=maximum), X_0=0
dupire_gradient_surface(strikes,maturities,calls,*,rate,dividend,variance_floor,vol_cap) - gradient twice; dict derivatives, raw_variance, local_vol
project_call_slice(strikes,calls,*,forward,discount) - nearest convex decreasing call slice within no-arbitrage bounds
localvol_paths(S0,times,*,rate,dividend,local_vol,normals) - log-Euler; callable local_vol(t,spots); caller's path-by-step normals; includes S0
barrier_digital_price(S,K,H,T,r,sigma,*,q,kind,payoff,direction,knock,cash=1) - continuous barrier binary; payoff cash|asset, direction up|down, knock in|out
gap_option_price(S,payoff_strike,trigger,T,r,sigma,*,q,kind) - gap call|put; payment may be negative
dupire_fd_surface(strikes,maturities,calls,*,rate,dividend,variance_floor,min_denominator,time_edge_order) - interior three-point curvature; missing/small denominator -> NaN

[mc_tools] unit-decimal vol/rate Greeks; per-year theta; caller's normal draws
gbm_greek_samples(S,K,T,r,sigma,*,q,kind,averaging,normals,theta) - price/pathwise/lr sample arrays; averaging 'terminal'|'arithmetic'; theta 'calendar'|'maturity'
gbm_fd_greek_samples(S,K,T,r,sigma,*,q,kind,averaging,normals,spot_bump,vol_bump,time_bump,rate_bump,theta) - central common-random-number Greek samples
mc_estimate(samples,*,pair_antithetic) - mean, variance, se; pair first/second halves before computing antithetic SE
correlated_gbm_terminal(spot,T,mu,sigma,*,correlation,normals,method) - multi-asset terminal prices; caller draws; method cholesky|eigh

[attribution] simple period returns; sectors on last axis, explicit cash, no multiperiod linking
brinson_fachler(portfolio_weights,benchmark_weights,portfolio_returns,benchmark_returns) - sector allocation/selection/interaction and period portfolio/benchmark/active returns
vol_target_weights(weights,covariance,*,target_vol,periods_per_year,max_scale) - full covariance risk scaling; cap on multiplier, no implicit gross normalization

[pricing_ext] explicit integration bounds, method and endpoint conventions
multifactor_heston_price(S,K,T,r,factors,*,q,kind,quadrature,bounds,points=256,limit=200,epsabs=1e-9,epsrel=1e-9) - independent CIR rows v0,kappa,theta,sigma_v,rho; quad|gauss_legendre
variance_replication(strikes,otm_prices,T,*,discount,endpoint) - static log contract; endpoint one_sided|trapezoid; dict variance,volatility,weights
synthetic_forward_bounds(strike,call_bid,call_ask,put_bid,put_ask,*,discount) - executable put-call parity forward lower/upper/midpoint using opposite quote sides
finite_difference_risk(pricer,parameters,*,steps,mixed) - deterministic scalar callable -> value, gradient, Hessian; explicit parameter-unit bumps

[tail_tools] losses positive; alpha confidence; explicit estimation conventions
weighted_var_es(losses,weights,alpha,*,quantile,es_rule) - quantile inverted_cdf|linear_cdf; ES fractional|ge|gt
gpd_observed_information(excesses,xi,beta,*,rel_step,max_halvings) - observed likelihood Hessian, covariance, se_xi, se_beta
normal_mixture_var_es(weights,means,scales,alpha,*,bounds,es_method) - scales are standard deviations; ES analytic|quad
kde_var_es(losses,alpha,*,bw_method,grid,epsabs) - KDE CDF-grid quantile and quadrature tail ES; explicit bandwidth and grid

[rate_tree] uniform time grid; curve valid one step beyond horizon for terminal rate calibration
hw_trinomial_tree(curve,a,sigma,*,dt,steps) - Euler-moment Hull-White tree, Arrow-Debreu curve fit; j_max optional
trinomial_rollback(tree,exercise_values,*,cashflows=None) - node-payoff optimal stopping; cashflow paid before exercise

[iv_tools] annual decimal sigma; no algorithm substitution on failure
iv_newton(price,S,K,T,r,*,kind,q,initial,tol,maxiter,criterion,vega_floor) - Newton inversion, status/residual/iterations; criterion 'step'|'price'
iv_li_atm(price,S,T,*,q) - Li forward-ATM cubic small root; arccos domain failure -> nan

[timeseries_tools] aligned chronological observations; explicit lag and sample-size conventions
lag_correlations(a,b,*,lags,min_pairs) - numpy.corrcoef(a[t],b[t+k]); positive k means a leads; pairwise finite
fisher_correlation_difference(rho_a,rho_b,*,n_a,n_b,clip,level) - independent-sample Fisher z diagnostic; no search/dependence correction

[execution] chronological complete days; integer allocations and fills; caller handles prices/fees
volume_share_profile(volume,*,method,half_life=None,quantile_method='linear',day_axis=0) - daily shares first; mean|median|ewma|winsorized; optional winsor_quantiles
profile_r2(actual,predicted,*,axis=-1,zero_sst='exact') - centered R2; constant exact=1 else0, or raise
largest_remainder(quantity,weights,*,tie_order=None) - Hamilton integer allocation, exact total, stable tie order
participation_fills(target,market_volume,*,participation,carry_unfilled=True) - scheduled integer releases with per-bar capacity and no lookahead

[arch_tools] actual installed arch dependency; delayed import; params retain raw fit units
arch_garch_fit(returns,*,mean,dist,input_scale,rescale,cov_type) - arch fit; risk outputs in input return units; vol/p/o/q/power/options optional
rolling_garch_var(returns,*,window,origins,alpha,horizon,horizon_method,mean,dist,input_scale,rescale,cov_type,on_failure) - past-window GARCH forecasts; sqrt_time explicit; failure raise|nan

[rates_ext] dated cashflows; explicit calendars, compounding, interpolation, risk bumps and calibration
timestamp_year_fraction(start,end,*,days_per_year) - elapsed seconds including fractions of days; timezone-aware values in UTC
dated_year_fraction(start,end,*,basis) - act/360|act/365f|act/365.25|30e/360|30/360_us|act/act_isda
dated_schedule(start,end,*,months,direction,stub,end_of_month,business_day,holidays) - unadjusted accrual dates and adjusted payments
DiscountCurve(times,values,*,input_type,comp,interp,extrapolate) - input zero|df; linear_zero|log_df|cubic_rt; flat_zero|flat_forward|raise
bootstrap_par_curve(maturities,rates,*,frequency,interp,extrapolate,bracket,xtol) - simultaneous par-instrument repricing on final curve
bootstrap_projection_curve(instruments,*,discount_curve,interp,extrapolate,bracket,xtol) - deposit/FRA/swap projection curve with separate discounting
floating_leg_cashflows(starts,ends,pay_times,accruals,*,projection_curve,notional,spread,fixings) - forward or known-fixing coupons; NaN marks unknown fixing
dual_curve_swap(fixed_times,fixed_accruals,float_starts,float_ends,float_times,float_accruals,*,discount_curve,projection_curve,fixed_rate,notional,spread,pay_fixed,fixings) - discounted fixed/floating legs and par rate
mtm_principal_cashflows(reset_times,notionals,*,final_time,initial_exchange,final_exchange) - signed reset exchanges; receive initial and increases, repay final
price_cashflows(times,amounts,*,curve,spread) - dated cashflow PV with continuous z-spread
curve_cashflow_risk(times,amounts,*,curve,comp,ytm) - duration/convexity in selected yield compounding
cashflow_ytm(price,times,amounts,*,comp,bracket,xtol) - bracketed yield root
z_spread(price,times,amounts,*,curve,bracket,xtol) - bracketed continuous spread root
key_rate_risk(price_function,curve,*,key_times,bump,scheme,shape) - zero-rate node|triangular key bumps; normalized risk needs positive base PV
quote_key_rate_risk(price_function,quotes,*,rebuild,bump,scheme) - quote bumps followed by full caller-defined curve rebuild
accrued_interest(settlement,previous_coupon,next_coupon,*,coupon_rate,face,frequency,basis,method) - day_count|period_fraction
deterministic_call_price_cap(noncall_price,call_times,call_prices,coupon_times,coupon_amounts,*,curve) - PV cap from dated calls; no stochastic exercise model
fx_forward(spot,base_df,quote_df) - quote currency per base unit; spot * base_df / quote_df
fx_implied_basis(spot,forward,base_df,quote_df,time,*,basis_leg,comp) - basis on explicit base|quote leg; continuous|simple
fit_parametric_curve(tau,yields,*,model,lam_form,method,x0,bounds,options,fixed_lams=None) - ns|svensson; explicit optimizer, success diagnostics; optional multistarts

[factors] finite aligned panels and explicit initialization; no hidden date shifting or whitening
ipca_fit(returns,Z_lagged,*,n_factors,F0,tol,max_iter,convergence,rotation,sign_rule) - ALS; rotation none|covariance, sign none|mean
dependent_sorts(first,second,ids,*,n_first,n_second) - stable IDs break ties; second ranks within first bucket
neutralize_weights(weights,exposures,*,include_intercept,target_exposure=None,net_target=None) - least-distance equality projection; default preserves net weight
turnover_cap(target,previous,*,max_turnover,convention) - interpolate weights to satisfy one_way|l1 turnover limit

[risk_ext] aligned standardized innovations; no marginal fitting, centering or implicit rescaling
dcc_filter(std_resid,*,a,b,qbar=None,q0=None,vol=None) - scalar DCC1,1; Q0=qbar, lagged innovations; returns Q,R,H and distinct likelihood conventions
dcc_fit(std_resid,*,qbar=None,q0=None,vol=None,method='L-BFGS-B',tol=1e-8,max_iter=1000) - positive stationary DCC QML in bounded logit coordinates; optional actual-coefficient start
stationary_bootstrap_indices(n_obs,*,n_samples,length,p,seed) - circular geometric blocks, common row indices for all assets, caller seed or Generator

[input_tools] explicit row calendar and missing policies; no I/O, units or estimation
numeric_time_frame(frame,*,date_col,value_cols,numeric_errors,missing) - sorted numeric DataFrame with unique naive date index; numeric_errors raise|coerce, missing raise|drop (joint finite rows)
date_row_position(index,date,*,direction) - sorted unique naive dates; direction next|previous|exact includes equality; no match raises IndexError
lagged_sample(values,*,lags,start,stop,missing) - y,X,positions dict; response interval [start,stop), positive row lags in column order; missing raise|drop AFTER pairing

labelled_panel(frame,*,time_col,entity_col,value_cols,times,entities,numeric_errors,missing) - explicit (T,N,L) values and labels; source_positions (T,N), finite_mask; typed ordered axes, duplicates/unknown raise; numeric_errors raise|coerce, missing raise|nan; caller aligns panels and lags

[streaming] fixed observation count; no RNG, resampling, precision reduction or task-specific computation
batch_slices(total,*,batch_size) - consecutive slices over exactly total rows, last partial batch; zero total yields none
OnlineMoments() - update(samples), merge(other), result(ddof=...) -> n,mean,variance,covariance,se arrays; rows=observations; O(d^2) state; pre-pair CRN differences/antithetic averages

[calendar_input] explicit calendar formats and row-loss reports; no I/O, sorting, deduplication or task policy
parse_calendar_dates(values,*,formats,missing='raise') - (dated Series,report); explicit strptime calendar formats; conflicting parses/invalid strings raise; missing raise|preserve; source order and duplicate indices retained
filter_rows(frame,keep,*,reason) - (filtered frame,report); exact boolean row mask; input/kept/dropped counts and positions relative to this input, no inferred rule

"""
# 0.5.0 appends its module blocks after the 0.4.0 CATALOG, which is its first CATALOG_040_LEN characters byte for
# byte; a reader that offers 0.4.0 only (the controller without T1_S12_LIB) cuts the CATALOG there.
CATALOG_040_LEN = 15393
CATALOG_050_LEN = 17893
