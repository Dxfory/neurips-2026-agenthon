"""Family notes for T1_ROUTE: the plan names the task's family (FAMILY: <label>) and the program prompt carries that
family's note. Each note is generic, written from the dev fast-path survey (2026-10-01): the library functions that
serve the family (by name), the method-choice rule (the method the TASK pins wins, exactly as written; only where the
TASK leaves it free does the note prefer a method), templates, identities and idioms, and the family's pitfalls. No
task names, ids, expected values or task constants. Frozen before any held-out run: a change is a new experiment."""
from __future__ import annotations

# (status slug, the label the plan writes, synonyms: patterns on the label's lower-cased words joined by single
# spaces). The labels are matched first, the synonyms only when no label occurs, and a label without synonyms
# (other) only when neither does; the earliest match wins.
FAMILIES = (
    ('closed_form', 'closed-form option pricing',
     (r'closed form', r'analytic\w* (?:option )?pric\w*', r'option pric\w*', r'black scholes', r'exotic options?')),
    ('numerical_engine', 'numerical pricing engine',
     (r'numerical', r'pdes?', r'finite differences?', r'lattices?', r'binomial', r'trinomial', r'crank nicolson',
      r'local vol\w*', r'monte carlo pric\w*')),
    ('process', 'stochastic-process fit and simulation',
     (r'stochastic process\w*', r'process fit\w*', r'sdes?', r'ornstein', r'ou', r'jump diffusions?',
      r'mean revert\w*', r'mean reversion', r'short rates?')),
    ('risk', 'risk/VaR/credit/copula',
     (r'risk', r'var', r'cvar', r'value at risk', r'expected shortfall', r'credit', r'copulas?', r'garch')),
    ('rates', 'rates/fixed income/FX',
     (r'rates?', r'fixed income', r'fx', r'yield curves?', r'curves?', r'bonds?', r'swaps?', r'currenc\w*',
      r'foreign exchange')),
    ('ledger', 'backtest ledger',
     (r'backtest\w*', r'ledgers?', r'trading strateg\w*', r'rebalanc\w*', r'attribution', r'hedg\w*', r'pnl')),
    ('econometrics', 'econometrics/factor models/estimators',
     (r'econometric\w*', r'factor models?', r'estimat\w*', r'regressions?', r'pca', r'event stud\w*',
      r'realized vol\w*', r'ols', r'time series', r'panel')),
    ('execution', 'execution/TCA/data audit',
     (r'execution', r'tca', r'transaction costs?', r'audit\w*', r'microstructure', r'order books?', r'vwap',
      r'twap', r'scheduling', r'participation')),
    ('wrangling', 'data wrangling/parsing',
     (r'wrangl\w*', r'pars\w*', r'data clean\w*', r'data prep\w*', r'cleaning', r'etl', r'extraction')),
    ('other', 'other', ()),
)
FAMILY_SLUGS = tuple(slug for slug, _, _ in FAMILIES)
LABELS = {slug: label for slug, label, _ in FAMILIES}

# The first line of every note.
NOTE_HEAD = 'The TASK always wins over this note, which guides only what the TASK leaves free.'
# The library line; only the functions the prompt's catalog lists are named.
LIBRARY_LINE = "- Library (qfkit, where it computes the TASK's own method): {names}."
# The four rules common to every family; the last line of every note.
COMMON = ('- Always: follow literally the TASK sentence that pins each graded method; pass conventions to library '
          'functions as explicit keywords; read library results only by their documented keys; checks of your own '
          'results print a warning, never raise.')

# slug: (library functions, ((item, the library functions it names), ...)). An item that names a function the
# prompt's catalog does not list is left out.
NOTES = {
    'closed_form': (
        ('bsm_greeks', 'implied_vol', 'barrier_price', 'lookback_fixed_price', 'asian_curran_price',
         'baw_critical_price', 'geske_compound_price', 'kirk_spread_price', 'cliquet_price'),
        (("A formula the TASK writes out wins; one it only names: its function, called once on whole arrays after "
          "checking each argument's meaning (leg or total maturity, running extreme so far, fixing times, kind and "
          "underlying).", ()),
         ("Printed identities: European put-call parity (American: bounds only; barriers: in + out = vanilla); "
          "the BSM PDE residual; compound call-on-call minus put-on-call = underlying call minus the discounted "
          "first strike.", ()),
         ("An MC check of a closed form: exact lognormal steps on payoff dates only, one draw set per maturity "
          "reused across strikes, same r and q, discounted. A seeded MC graded against a reference keeps the "
          "TASK's layout.", ()),
         ("Greek units (theta per year or day, vega per unit or percent) as the TASK states.", ()))),
    'numerical_engine': (
        ('fd_solve', 'richardson', 'crr_price', 'gbm_mc_price', 'heston_price'),
        (("Keep the scheme, grid, boundaries and exercise method the TASK pins; a library engine only if it "
          "covers every TASK feature (cash dividends, exercise style); price and Greeks from one method and "
          "grid.", ()),
         ("European steps: one banded solve per step (scipy.linalg.solve_banded), no dense inverse. American: "
          "PSOR as the TASK pins it (nodes in order or red-black, never a simultaneous update with omega > 1), "
          "warm-started from the last step, or Brennan-Schwartz with its sweep matching one exercise region. Cash "
          "dividend D at t: V(S,t-) = V(S-D,t+).", ()),
         ("MC bump Greeks: one pricer that reseeds its generator on every call (common random numbers), "
          "vectorised over paths, drawing in the TASK's layout and order when seeded.", ()),
         ("Local vol from total implied variance with analytic smile derivatives; dupire_local_vol only where the "
          "TASK pins no stencil.", ('dupire_local_vol',)))),
    'process': (
        ('cir_logpdf', 'vasicek_logpdf', 'merton_logpdf', 'ou_fit', 'ou_simulate', 'ou_moments', 'detect_jumps',
         'multistart_minimize'),
        (("Likelihood: the transition log-density in ONE array call on (x[1:], x[:-1]) inside the objective, never "
          "per observation; spend the time saved on more optimiser starts.", ()),
         ("Fit the variant the TASK's formula describes: ou_fit method='euler_diff' and 'exact_ar1' give different "
          "parameters.", ('ou_fit',)),
         ("When the jumps enter the mean-reverting SDE (ou_jump_moments), the jump variance mean-reverts too; in "
          "a Merton jump-diffusion it accumulates.", ('ou_jump_moments',)),
         ("Simulate with the TASK's scheme, recording convention, seed and draw order; vectorise across paths, "
          "never across time in a recursion.", ()),
         ("An author's name can span models (short rate or credit, jump-diffusion or structural): use only the "
          "model the TASK means.", ()),
         ("Printed checks: MC mean and variance within a few standard errors of the analytic moments.", ()))),
    'risk': (
        ('var_es_historical', 'var_es_forecast', 'rolling_var_backtest', 'kupiec_pof', 'garch11_fit',
         'copula_sample'),
        (("var_es_historical and the other library VaR/ES functions take LOSSES (pass -returns); quantile and ES "
          "rule as the TASK defines.", ('var_es_historical',)),
         ("GARCH: estimator, mean, distribution and scaling as the TASK pins; check whether a volatility is in "
          "return or percent units.", ()),
         ("Portfolio or credit MC: one parametrised simulator; ES contributions are per-position tail "
          "means of the SAME simulation (they sum to ES); check a discrete-loss VaR/ES against the exact "
          "conditional distribution where one exists.", ()),
         ("Pinned seed: the TASK's generator and draw order exactly (np.random.seed and default_rng streams "
          "differ).", ()),
         ("Compound Poisson loss: pmf = ifft(exp(lam*(fft(f)-1))) on a grid several times the target quantile "
          "(other frequencies: their pgf of fft(f)); MC: seed once, block draws, np.bincount. A t copula needs "
          "nu=; guard tau <= 0.", ()))),
    'rates': (
        ('ZeroCurve', 'bootstrap_zero_curve', 'bond_price_curve', 'bond_risk', 'fit_ns', 'par_swap_rate'),
        (("Write ONE value(state); every risk and stress revalues it on the shocked object the TASK names "
          "(quotes, zero rates, a basis node, the valuation date), with the TASK's shock size, sign and "
          "differencing.", ()),
         ("Many instruments: a cash-flow matrix (instruments x dates) times the discount vector for every price "
          "and risk.", ()),
         ("Where the TASK prints no bootstrap formula: closed form only for single-payment pillars, a 1-D root "
          "solve for a pillar feeding interpolated forwards. A par rate or spread linear in the value: closed form. "
          "Nelson-Siegel with fixed lambda: linear least squares.", ()),
         ("FX: keep directly quoted crosses; derive others from one common-currency bid/ask table, taking opposite "
          "sides; covered interest parity with each currency's day count.", ()),
         ("Debug-and-fix tasks: assume planted bugs; patch once per rule the TASK states.", ()))),
    'ledger': (
        ('signal_trade_ledger', 'weights_ledger', 'delta_hedge_pnl', 'bsm_escrowed_dividend', 'max_drawdown'),
        (("Map every stated rule (signal timing, execution price, sizing, fees, stop, forced close, pending "
          "orders, exit priority) to an explicit keyword or line; where the TASK is silent (same-bar re-entry, "
          "lag, warm-up), decide from its wording and print the choice.", ()),
         ("Trade reports come from the ledger's own per-trade rows; never re-pair buys and sells.", ()),
         ("Event ledgers: vectorise only cash-independent columns; quantity and every quantity-scaled field inside "
          "the time loop.", ()),
         ("Month-ends: resample('ME') on the full history keyed by Period, then slice; recursive filters: "
          "the TASK's seed (.ewm(adjust=False) from the first valid value only when it gives none).", ()),
         ("delta_hedge_pnl: pass t_remaining explicitly.", ('delta_hedge_pnl',)),
         ("Printed checks: equity = cash + holdings on every row; summed trade P&L = final minus initial "
          "equity.", ()))),
    'econometrics': (
        ('ols', 'newey_west_lags', 'rolling_ols', 'fama_macbeth', 'grs_test', 'event_study', 'event_study_tests',
         'realized_measures', 'noise_variance'),
        (("A formula the TASK writes out wins; an estimator it only names: its standard textbook form, printed as "
          "your assumption; no hand-written fallback around a library call.", ()),
         ("Residuals = y - X @ b; one multi-RHS least-squares solve per date when every regression uses the same "
          "rows; projection w - X @ (pinv(X) @ w); an expanding Fama-MacBeth forecast is the running mean of slopes "
          "computed once.", ()),
         ("Standardise cross-sections with groupby().transform; write each repeated analysis once as a function "
          "called per sample.", ()),
         ("HAC lags by the rule the TASK states (newey_west_lags rule=).", ('newey_west_lags',)),
         ("HMM: scaled forward-backward recursions; an iterative fit with a rotation or scale freedom stops on the "
          "objective's change, not on raw parameters.", ()))),
    'execution': (
        (),
        (("Build each grid (buckets, sessions, schedules) from the window and step the TASK states (np.arange "
          "half-open, pd.date_range), not from the rows present; empty buckets stay rows (reindex to the full "
          "grid); a calendar the TASK supplies filters rows instead.", ()),
         ("As-of on the full sorted array, never cut first, no fallback row when none exists, on the TASK's side: "
          "first at or after t is np.searchsorted(times, t, side='left'), last at or before t side='right' less "
          "one.", ()),
         ("Aggregate with np.bincount or groupby reindexed to the grid; days x bins matrices; decay weights in the "
          "window's row order (oldest first: the newest row weighs most).", ()),
         ("Integer allocation: largest remainder with a stable sort; pair records with merge on keys; medians "
          "with np.median.", ()),
         ("Year fractions from exact timestamps (total_seconds) when times carry a time of day.", ()),
         ("Printed checks: grid size, empty buckets present, the schedule sums to the total.", ()))),
    'wrangling': (
        (),
        (("Parse each stated date format explicitly (format=); format='mixed' only where no day-first numeric "
          "format occurs; errors='coerce' only where the TASK drops unparseable values; then check NaT and row "
          "counts.", ()),
         ("Count every dropped, filled or corrected row and report it as the TASK asks; no silent coerce or "
          "fill.", ()),
         ("Keep raw inputs immutable: compute per-row factors as one vector and apply them once.", ()),
         ("Vendor sheets: read with header=None, find the header row by its labels, one number parser for every "
          "cell (thousands separators, parenthesised negatives, percent signs).", ()),
         ("Generate patterned keys from base lists; one tidy long table, everything by groupby.", ()))),
    'other': ((), ()),
}
