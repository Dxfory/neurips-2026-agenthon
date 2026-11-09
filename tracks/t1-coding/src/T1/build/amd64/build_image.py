#!/usr/bin/env python3
"""Build the AMD64 candidate from hash-verified official PyPI wheels.

This candidate recipe keeps the ARM64 Dockerfile unchanged. The build RUN
steps have no network. Only explicit controller sources and verified dependency
files enter the temporary tar context; official task files never enter it.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from t1_phase1 import solver_profile as profiles


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pins(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)', line)
        if not match:
            raise ValueError('Offline requirements must contain exact version pins only')
        name = re.sub(r'[-_.]+', '-', match[1].lower())
        if name in result:
            raise ValueError('Duplicate requirement')
        result[name] = match[2]
    return result


def contract_problems(image, solver_profile=None):
    """IMAGE-SUBMISSIONS.md: linux/amd64 and the qfbench2.interface_version=2.0 label."""
    problems = []
    if image.get('Architecture') != 'amd64' or image.get('Os') != 'linux':
        problems.append('platform')
    labels = (image.get('Config') or {}).get('Labels') or {}
    if labels.get('qfbench2.interface_version') != '2.0':
        problems.append('interface_label')
    if solver_profile is not None:
        expected = profiles.profile_env(solver_profile)
        if labels.get('t1.solver_profile') != solver_profile:
            problems.append('solver_profile')
        values = (image.get('Config') or {}).get('Env') or []
        pairs = [item.split('=', 1) for item in values if isinstance(item, str) and '=' in item]
        env = dict(pairs)
        if len(env) != len(pairs) or any(env.get(k) != v for k, v in expected.items()):
            problems.append('solver_flags')
    return problems


def profile_recipe(text, name):
    """Render an explicitly selected profile without changing either tracked recipe."""
    for key, value in profiles.profile_env(name).items():
        text, count = re.subn(r'\b'+re.escape(key)+r'=[^\s\\]+', key+'='+value, text)
        if count != 1:
            raise ValueError('Recipe is missing an unambiguous solver default')
    text, count = re.subn(r't1\.solver_profile="[^"]+"', 't1.solver_profile="'+name+'"', text)
    if count != 1:
        raise ValueError('Recipe is missing solver profile label')
    return text


def recipe_default_problems(text):
    """The registered run parameters whose tracked recipe line is not their image default (solver_profile.
    THIN_RUN_PARAM_DEFAULTS, written as recipe_value). The build refuses such a recipe instead of rendering over it,
    so a default changes only in solver_profile and both recipes together (scripts/thin_defaults.py)."""
    return [name for name, value in profiles.THIN_RUN_PARAM_DEFAULTS.items()
            if re.findall(r'(?m)^\s*' + re.escape(name) + r'=(\S*)', text) != [profiles.recipe_value(value)]]


def git_source_commit(root):
    try:
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5, check=False)
        value = result.stdout.strip()
        return value if result.returncode == 0 and re.fullmatch(r'[0-9a-f]{40}', value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_matches_commit(files, root, commit):
    """Compare only actual build inputs, not unrelated histories or virtual environments."""
    try:
        top = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
                             capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        hashes = {}
        for name, path in files.items():
            path = Path(path)
            if path.is_symlink():
                return False
            relative = path.resolve().relative_to(Path(top).resolve()).as_posix()
            raw = subprocess.run(['git', '-C', top, 'show', commit+':'+relative],
                                 capture_output=True, timeout=5, check=False)
            observed = path.read_bytes()
            if raw.returncode != 0 or raw.stdout != observed:
                return None
            hashes[name] = hashlib.sha256(observed).hexdigest()
        return hashes
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def build_provenance(files, root=ROOT):
    commit = git_source_commit(root)
    hashes = source_matches_commit(files, root, commit) if commit else None
    return {'source_commit': commit, 'source_state': 'frozen' if hashes else 'unfrozen',
            'verified_source_sha256': hashes if isinstance(hashes, dict) else {}}


def read_bound_file(path, expected):
    if path.is_symlink():
        raise ValueError('Symlink in explicit build context')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Source changed after matching the source commit')
    return raw


def context_binding(hashes):
    # The generated metadata cannot hash itself. The full external manifest also
    # hashes that file, so its content remains independently bound by the report.
    values = {k:v for k,v in hashes.items() if k != 't1_phase1/build_identity.py'}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def add_bound_file(archive, name, path, expected):
    """Hash the bytes actually entering the tar, not merely a preceding disk read."""
    if path.is_symlink():
        raise ValueError('Symlink in explicit build context')
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o644
    seen = hashlib.sha256()
    with path.open('rb') as stream:
        class Reader:
            def read(self, size=-1):
                data = stream.read(size)
                seen.update(data)
                return data
        archive.addfile(info, Reader())
        if stream.read(1) or seen.hexdigest() != expected:
            raise ValueError('Build context changed after its manifest was frozen')


def identity_smoke(context, image_id, name, provenance):
    script = '''import json
from pathlib import Path
from t1_phase1.solver_profile import identity
from t1_phase1 import thin
import qfkit
status = {}
context = thin.task_context(Path('/input'), Path('/app/output'), 'Write /app/output/result.json.',
                            {'files': []}, '', (), status)
prompt = thin.plan_prompt(context)
api = thin.api_context('APIS: qfkit.bsm_price')
print(json.dumps({'identity': identity(), 'prompt': {
    'controller': 'thin', 'task_context_in_plan': context in prompt,
    'full_catalog_absent': all(qfkit.CATALOG.strip() not in text for text in (context, prompt, api)),
    'selected_api_present': 'bsm_price(' in api and 'kind' in api,
    'unrelated_api_absent': 'labelled_panel(' not in api}}))
'''
    command = ['docker', '--context', context, 'run', '--rm', '--network=none', '--read-only',
               '--user=65534:65534', image_id, 'python', '-c', script]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        packet = json.loads(done.stdout) if len(done.stdout) <= 8192 else {}
        data = profiles.safe_identity(packet.get('identity')) if isinstance(packet, dict) else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'ok': False}
    expected = profiles.identity(profiles.profile_env(name), sources={
        'qfkit': ROOT/'qfkit/qfkit/__init__.py', 'qfruntime': ROOT/'qfruntime/qfruntime/__init__.py'},
        provenance=provenance)
    prompt_expected = {'controller': 'thin', 'task_context_in_plan': True,
                       'full_catalog_absent': True, 'selected_api_present': True,
                       'unrelated_api_absent': True}
    prompt = packet.get('prompt') if isinstance(packet, dict) else None
    prompt_ok = (isinstance(prompt, dict) and set(prompt) == set(prompt_expected) and
                 all(type(prompt[k]) is type(v) and prompt[k] == v for k,v in prompt_expected.items()))
    return {'ok': done.returncode == 0 and data == expected and data.get('source_state') == 'frozen' and profiles.startup_problem(data) is None and prompt_ok,
            'identity': data, 'first_prompt_profile_verified': prompt_ok}


def solve_smoke(context, image_id):
    """Invoke the platform verb through PATH and its shebang, offline, without House."""
    command = ['docker', '--context', context, 'run', '--rm', '--network=none', '--read-only',
               '--user=65534:65534', image_id, 'solve', '--help']
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'returncode': None, 'error': type(exc).__name__}
    return {'ok': completed.returncode == 0 and 'usage: solve' in (completed.stdout or ''),
            'returncode': completed.returncode}


# One known value per qfkit module on the image's pinned numpy, scipy, pandas and statsmodels (qfkit's own suite
# ran on the host's): closed forms or a textbook example, not values qfkit produced.
# Handwritten label/position sentinels; no official task inputs or estimator assumptions.
LABELLED_PANEL_SMOKE = r'''
import numpy as np
import pandas as pd
import qfkit
assert qfkit.__version__ == "0.6.2"
raw_panel = pd.DataFrame({"t":["late","early","early"], "id":["b","a","b"],
                          "v":[32.,11.,12.], "w":[132.,111.,112.]}, index=[9,9,4])
raw_copy = raw_panel.copy(deep=True)
panel = qfkit.labelled_panel(raw_panel, time_col="t", entity_col="id", value_cols=["w","v"],
                            times=["early","late"], entities=["b","a"],
                            numeric_errors="raise", missing="nan")
assert panel["values"].shape == (2,2,2)
assert np.array_equal(panel["values"][0], [[112.,12.],[111.,11.]])
assert np.array_equal(panel["values"][1,0], [132.,32.])
assert np.isnan(panel["values"][1,1]).all()
assert np.array_equal(panel["source_positions"], [[2,1],[0,-1]])
assert panel["time_labels"] == ("early","late") and panel["entity_labels"] == ("b","a")
assert panel["feature_labels"] == ("w","v")
assert np.array_equal(panel["finite_mask"], [[[True,True],[True,True]],[[True,True],[False,False]]])
panel["values"][0,0,0] = 999.
assert raw_panel.equals(raw_copy)
for unit in ("ns", "D"):
    try:
        qfkit.labelled_panel(pd.DataFrame({"t":[1],"id":["a"],"v":[3.]}),
                            time_col="t",entity_col="id",value_cols=["v"],
                            times=[np.timedelta64(1,unit)],entities=["a"],
                            numeric_errors="raise",missing="nan")
    except ValueError:
        pass
    else:
        raise AssertionError("duration must not alias an integer label")
'''

STREAMING_SMOKE = r'''
# Fixed count coverage and analytic sample moments; no task-specific values.
assert [(s.start, s.stop) for s in qfkit.batch_slices(7, batch_size=3)] == [(0,3),(3,6),(6,7)]
stream_stats = qfkit.OnlineMoments().update([[1.,2.],[2.,4.]]).update([[3.,6.]])
stream_result = stream_stats.result(ddof=1)
assert stream_result["n"] == 3
assert np.allclose(stream_result["mean"], [2.,4.])
assert np.allclose(stream_result["covariance"], [[1.,2.],[2.,4.]])
assert np.allclose(stream_result["se"], np.sqrt([1./3.,4./3.]))
'''

CALENDAR_INPUT_SMOKE = r'''
# Handwritten mixed-format dates and exact row positions; no inferred calendar policy.
calendar_raw = pd.Series(["2024-05-01", "05/02/2024", None], index=[7,7,9])
calendar_dates, calendar_report = qfkit.parse_calendar_dates(
    calendar_raw, formats=["%Y-%m-%d", "%m/%d/%Y"], missing="preserve")
assert calendar_dates.iloc[:2].dt.strftime("%Y-%m-%d").tolist() == ["2024-05-01", "2024-05-02"]
assert pd.isna(calendar_dates.iloc[2]) and calendar_dates.index.tolist() == [7,7,9]
assert calendar_report["rows_input"] == calendar_report["rows_output"] == 3
assert calendar_report["rows_parsed"] == 2 and calendar_report["missing_positions"] == [2]
calendar_kept, calendar_filter = qfkit.filter_rows(
    pd.DataFrame({"value":[4,5,6]}, index=[7,7,9]), [True,False,True], reason="declared fixture mask")
assert calendar_kept["value"].tolist() == [4,6]
assert calendar_filter["rows_input"] == 3 and calendar_filter["rows_kept"] == 2
assert calendar_filter["dropped_source_positions"] == [1]
'''

LIBRARY_SMOKE = r'''
import numpy as np
import qfkit
assert abs(qfkit.bsm_price(100, 100, 1, 0.05, 0.2, kind="call") - 10.450583572185565) < 1e-9
assert abs(qfkit.bond_ytm(98.3850627729, 2, 0.06, freq=2, comp="continuous") - 0.0676243872) < 1e-9  # Hull
fit = qfkit.ols([1, 3, 2, 5, 4], [1, 2, 3, 4, 5], add_const=True)  # slope 8/10, its t(3) two-sided p-value
assert np.allclose(fit["params"], [0.6, 0.8], rtol=0, atol=1e-12)
assert abs(fit["pvalues"][1] - 0.10408803866182792) < 1e-9
assert abs(qfkit.kupiec_pof(5, 250, 0.01)["lr_uc"] - 1.956809788230622) < 1e-9
assert np.allclose(qfkit.var_es_historical(np.arange(1.0, 101.0), 0.95, method="linear", es_rule="ge"),
                   [95.05, 98.0], rtol=0, atol=1e-9)
assert abs(qfkit.vasicek_bond_price(0.03, 5, 0.5, 0.04, 0.01) - 0.8342873600428864) < 1e-12
# Margrabe with a riskless asset 2 (sigma2=0, S2=K e^-rT) is the BSM call above.
assert abs(qfkit.margrabe_price(100, 100 * np.exp(-0.05), 1, 0.2, 0.0, 0.0) - 10.450583572185565) < 1e-9
# Hull's 5-step American put (S=K=50, r=10%, sigma=40%, T=5 months; printed 4.49), rolled back by hand.
assert abs(qfkit.crr_price(50, 50, 5 / 12, 0.1, 0.4, kind="put", exercise="american", steps=5)
           - 4.488458534725916) < 1e-9
# Lewis (2000) Heston reference call, K=100.
assert abs(qfkit.heston_price(100, 100, 1, 0.01, 0.04, 4.0, 0.25, 1.0, -0.5, kind="call", q=0.02)
           - 16.070154917028834) < 1e-8
# Jamshidian's Hull-White zero-bond put on a flat 5% curve: K P(3) N(sp-h) - P(9) N(-h), sp = 0.0677676.
assert abs(qfkit.hw_bond_option(0.74, 3, 9, 0.1, 0.01, curve=lambda t: np.exp(-0.05 * t), kind="put")
           - 0.016875845431760295) < 1e-12
merton = qfkit.merton_structural(100, 100, 1, 0.05, 0.2)  # equity is the BSM call; PD = N(-d2) = N(-0.15)
assert abs(merton["equity"] - 10.450583572185565) < 1e-9 and abs(merton["pd_rn"] - 0.4403823076297575) < 1e-12
# Hull's GARCH(1,1) omega=2e-6, alpha=0.13, beta=0.86: V_L = 0.0002, day 10 = V_L + 0.99^9 (3e-4 - V_L).
assert abs(qfkit.garch11_forecast(2e-6, 0.13, 0.86, var_next=3e-4, horizon=10)["variance"][9]
           - 0.0002913517247483641) < 1e-15
t = np.arange(80.0)  # Engle-Granger through statsmodels: the no-lag Dickey-Fuller t of the OLS residuals
x = np.cumsum(np.sin(1.3 * t))
eg = qfkit.engle_granger(1 + 2 * x + 0.3 * np.sin(2.1 * t + 0.5), x, trend="c", maxlag=0, autolag=None)
assert abs(eg["stat"] - -15.306858795493973) < 1e-8
# Bodie-Kane-Marcus two-asset minimum variance (sd 12%/20%, rho 0.3): w_D = (0.04-0.0072)/0.04 = 0.82.
assert np.allclose(qfkit.min_variance_weights([[0.0144, 0.0072], [0.0072, 0.04]], long_only=False),
                   [0.82, 0.18], rtol=0, atol=1e-12)
# Sheppard: a Gaussian copula at (1/2, 1/2) is 1/4 + asin(rho)/(2 pi), 1/3 for rho = 1/2.
assert abs(qfkit.copula_cdf([0.5, 0.5], 0.5, family="gaussian") - 1 / 3) < 1e-14
# A forward start with q = 0 is the BSM call above whatever its start: e^{-q t} BSM(S, alpha S, T - t).
assert abs(qfkit.forward_start_price(100, 0.5, 1.5, 0.05, 0.2, kind="call", alpha=1.0)
           - 10.450583572185565) < 1e-9
# qfkit 0.4.0 [hedging]: a dividend of 5 in half a year on a spot of 100 + 5 e^{-0.025} leaves S_eff = 100 (BSM call).
assert abs(qfkit.bsm_escrowed_dividend(100 + 5 * np.exp(-0.025), 100, 1, 0.05, 0.2, kind="call",
                                       dividends=[(0.5, 5.0)])["price"] - 10.450583572185565) < 1e-9
# qfkit 0.4.0 [pde]: Crank-Nicolson on dS = 1 is within 3e-3 of that BSM call.
assert abs(qfkit.fd_solve(100, 100, 1, 0.05, 0.2, kind="call", exercise="european", scheme="cn", s_max=400,
                          n_space=400, n_time=200, boundary="dirichlet", american_method=None, tol=None, omega=None,
                          exercise_tol=None, return_grid=False)["price"] - 10.450583572185565) < 3e-3
# qfkit 0.4.0 [realized]: 5-minute on-grid RV by hand: 09:31 is off the grid, the log returns 0.01 and -0.03 give
# 0.01^2 + 0.03^2.
import pandas as pd
day = pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:31", "2024-01-02 09:35", "2024-01-02 09:40"])
rm = qfkit.realized_measures(pd.Series(100 * np.exp([0.0, 0.05, 0.01, -0.02]), index=day), grids=(5,),
                             anchor="09:30", rule="on_grid", noise_estimator="bandi_russell", correction="none",
                             omega_source="per_day", bv_scaling="none")
assert abs(rm["table"]["rv_5"].iloc[0] - 1e-3) < 1e-15 and rm["table"]["n_5"].iloc[0] == 2
# qfkit 0.4.0 [ou]: a noiseless AR(1) x_t = 1 + 0.5^t: slope b = 1/2 exactly, so theta = 1 and the half-life is 1 period.
fit = qfkit.ou_fit(1 + 0.5 ** np.arange(12), dt=1.0, method="exact_ar1", sigma_method="exact", resid_ddof=0)
assert abs(fit["half_life_periods"] - 1) < 1e-12 and abs(fit["theta"] - 1) < 1e-12
# qfkit 0.4.0 [ledger]: a constant 50/50 mix earns 0.5*0.10 + 0.5*0.02 = 0.06, less 10 bps of the initial build.
assert abs(qfkit.weights_ledger(np.array([[0.10, 0.02]]), [0.5, 0.5], lag=0, rebalance="every_period", drift=False,
                                cost_bps=10, cost_base="abs_dw", cost_timing="effective")["returns"].iloc[0]
           - 0.059) < 1e-15
# qfkit 0.4.0 [events]: market-adjusted abnormal returns are R - R_m: 0.010 - 0.004 on each of 3 event days, CAR 0.018.
d = pd.bdate_range("2024-01-01", periods=12)
es = qfkit.event_study(pd.DataFrame({"X": np.full(12, 0.010)}, index=d), pd.Series(np.full(12, 0.004), index=d),
                       [("X", d[10])], estimation_window=(-10, -2), event_window=(-1, 1), model="market_adjusted",
                       min_est_obs=3, require_full_event_window=True, event_day_rule="exact")
assert abs(es["car"].iloc[0] - 0.018) < 1e-12
# qfkit 0.4.0 [varbt]: losses -1, 1 have mean 0 and population sd 1, so the normal VaR/ES are z_0.99 and
# phi(z_0.99)/0.01 (QRM 2.3).
fc = qfkit.var_es_forecast([-1.0, 1.0], 0.99, method="normal", quantile_method=None, es_rule=None, mu_sign=1, ddof=0)
assert abs(fc["var"] - 2.3263478740408408) < 1e-12 and abs(fc["es"] - 2.665214220345806) < 1e-12
# qfkit 0.5.0 [hmm]: one observation's likelihood is the pi-weighted mixture of the state normals.
mix = sum(p * np.exp(-(0.3 - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v) for p, m, v in ((0.6, -0.4, 0.5),
                                                                                           (0.4, 0.6, 0.2)))
post = qfkit.hmm_posterior([0.3], mu=[-0.4, 0.6], var=[0.5, 0.2], trans=[[0.8, 0.2], [0.3, 0.7]], pi=[0.6, 0.4])
assert abs(post["loglik"] - np.log(mix)) < 1e-12
# qfkit 0.5.0 [signals]: span 3 gives alpha 1/2, so the EMA of 1, 2, 3 from the first value is 1, 1.5, 2.25.
assert np.allclose(qfkit.ema([1.0, 2.0, 3.0], span=3, init="first"), [1.0, 1.5, 2.25], rtol=0, atol=1e-15)
# qfkit 0.5.0 [spectral]: Marchenko-Pastur upper edge (1 + sqrt(100/500))^2.
assert abs(qfkit.marchenko_pastur_edge(500, 100, sigma2=1.0)["lambda_max"] - (1 + np.sqrt(0.2)) ** 2) < 1e-14
# qfkit 0.5.0 [creditpf]: BCBS (2005) explanatory note, corporate PD 1%, LGD 45%, M 2.5: risk weight 92.32%.
w = (1 - np.exp(-0.5)) / (1 - np.exp(-50))
b = (0.11852 - 0.05478 * np.log(0.01)) ** 2
k = (qfkit.vasicek_loss_quantile(0.999, 0.01, lgd=0.45, rho=0.12 * w + 0.24 * (1 - w)) - 0.0045) / (1 - 1.5 * b)
assert abs(12.5 * k - 0.9232) < 5e-5
# qfkit 0.5.0 [copfit]: Akaike 2k - 2 log L and Schwarz k ln n - 2 log L.
crit = qfkit.information_criteria(-10.0, n=100, n_params=2)
assert crit["aic"] == 24.0 and abs(crit["bic"] - (2 * np.log(100) + 20)) < 1e-12
# qfkit 0.5.0 [aggloss]: a severity of exactly one unit makes the aggregate the Poisson(2) count itself.
g = qfkit.panjer_recursion([0.0, 1.0, 0.0, 0.0, 0.0], freq=("poisson", 2.0))
assert np.allclose(g, [np.exp(-2) * 2.0 ** n / [1, 1, 2, 6, 24][n] for n in range(5)], rtol=0, atol=1e-15)
# qfkit 0.5.0 [svi]: w(m) = a + b sigma at k = m.
assert abs(qfkit.svi_total_variance(0.0, 0.04, 0.4, -0.4, 0.0, 0.1) - 0.08) < 1e-15
# 0.6.0: independent probability/attribution identities and hand-computed quadrature.
assert qfkit.brownian_hit_cdf(0., 1., .03, .2, direction="up") == 1.
assert qfkit.mc_estimate([1., 3., 3., 1.], pair_antithetic=True)["se"] == 0.
bf = qfkit.brinson_fachler([.7,.3], [1.,0.], [.12,.003], [.1,.003])
assert abs(bf["active_return"] - (-.0151)) < 1e-12
assert abs(qfkit.variance_replication([1.,2.], [1.,4.], 1., discount=1., endpoint="trapezoid")["variance"] - 2.) < 1e-12
assert abs(qfkit.weighted_var_es([1.,3.,8.], [.2,.5,.3], .6, quantile="inverted_cdf", es_rule="fractional")["es"] - 6.75) < 1e-12
tree = qfkit.hw_trinomial_tree(lambda t: np.exp(-.03*np.asarray(t)), .1, .01, dt=.1, steps=10)
assert abs(qfkit.trinomial_rollback(tree, {}, cashflows={10:1.})["value"]-np.exp(-.03)) < 1e-12
assert qfkit.iv_li_atm(0., 100., 1., q=0.) == 0.
assert abs(qfkit.lag_correlations([1.,3.,2.,4.], [0.,1.,3.,2.], lags=[1], min_pairs=3)["correlation"][0]-1.) < 1e-12
assert np.array_equal(qfkit.largest_remainder(7, [1.,1.,1.], tie_order=[30,10,20]), [2,3,2])
sample = .01*np.sin(np.arange(120.))*(1.+.1*np.cos(np.arange(120.)/7))
archfit = qfkit.arch_garch_fit(sample, mean="Zero", dist="normal", input_scale=100., rescale=False, cov_type="robust")
assert np.allclose(archfit["residuals"], sample, rtol=0, atol=1e-14) and archfit["next_variance"] > 0.
assert abs(qfkit.fx_forward(1.2, .98, .96)-1.225) < 1e-14
assert np.allclose(qfkit.turnover_cap([.8,.2],[.5,.5],max_turnover=.15,convention="one_way"), [.65,.35])
dcc = qfkit.dcc_filter([[1.,2.],[2.,-1.],[-1.,.5]], a=.2, b=.3, qbar=np.eye(2))
assert np.allclose(dcc["Q"][1], [[1.,.4],[.4,1.6]], rtol=0, atol=1e-14)
# Explicit numeric row policy: preserve dates while sorting and coercing only declared value columns.
rows = qfkit.numeric_time_frame(pd.DataFrame({"day":["2024-01-02","2024-01-01"],"x":["3","1"]}),
                                date_col="day", value_cols=["x"], numeric_errors="raise", missing="raise")
assert rows["x"].tolist() == [1.,3.]
assert qfkit.date_row_position(rows.index, "2024-01-01", direction="exact") == 0
''' + LABELLED_PANEL_SMOKE + STREAMING_SMOKE + CALENDAR_INPUT_SMOKE + 'print("library-ok")\n'

RUNTIME_SMOKE = r'''
from qfruntime import Checks, StageCache, probe, EvidenceLedger, availability_filter, asof_vintages
import tempfile
checks = Checks(report_path="")
assert checks.contract("shape-unit", [1.,2.], shape=(2,), actual_unit="USD", expected_unit="USD")
assert checks.balance("sum", [1.,2.], 3., atol=0., rtol=0.)
assert all(row["status"] == "passed" for row in checks.finish()["checks"])
with tempfile.TemporaryDirectory() as temporary:
    cache = StageCache(temporary, namespace="synthetic-smoke")
    def validate(value):
        return value == {"sum":3}
    def compute():
        return {"sum":3}
    for attempt in range(2):
        assert cache.run("sum", version="1", inputs=[1,2], parameters={}, dependencies=[],
                         compute=compute, validate=validate) == {"sum":3}
    assert cache.stats()["computed"] == 1 and cache.stats()["hits"] == 1
small = probe(lambda n: sum(range(n)), sample_size=3, target_size=6, time_budget=60., memory_budget=1024)
assert small["result"] == 3 and small["target_executed"] is False
assert small["memory_status"] == "unverified" and small["fits_budget"] is None
ledger = EvidenceLedger()
record = ledger.add("cash", 12, unit="USD", period="2025", source_text="Cash: 12",
                    source_id="synthetic", start=0, end=8)
assert record["source"]["quote"] == "Cash: 12" and ledger.report()["conflicts"] == []
versions = [{"key":"cash", "value":12, "available_at":"2025-01-01T00:00:00Z",
             "revision_at":"2025-01-01T00:00:00Z"},
            {"key":"cash", "value":13, "available_at":"2025-01-01T00:00:00Z",
             "revision_at":"2025-03-01T00:00:00Z"}]
assert availability_filter(versions, as_of="2025-02-01T00:00:00Z") == versions[:1]
assert asof_vintages(versions, as_of="2025-02-01T00:00:00Z", key_fields=["key"])[0]["value"] == 12
from qfruntime import (extract_task_spec, enforce_contract, verify_quantity, ResourceBudget,
                      measure_resources, run_batches, Stage, Workflow, QuantitySpec,
                      read_document, extract_quantity)
from pathlib import Path
spec = extract_task_spec('quantity x: unit=fraction; shape=[]; axes=[]; method=analytic; time_basis=static; atol=0.000001; rtol=0')
quantity = spec.get('x')
assert enforce_contract(checks, 'defined-x', 3., expected=quantity, observed=quantity)['status'] == 'passed'
assert verify_quantity(checks, 'derivative-x', quantity=quantity, actual=3., operation='derivative',
                       inputs={'function':lambda x:x*x,'x':1.5})['status'] == 'passed'
measured = measure_resources(lambda:sum(range(100)))
assert measured['result'] == 4950 and measured['rss_status'] == 'measured'
batches = run_batches(lambda n,rng:[2.]*n, max_samples=10, batch_size=3, seed=4,
                      budget=ResourceBudget(wall_seconds=10))
assert batches['status'] == 'completed' and batches['samples'] == 10 and batches['mean'] == 2.
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary).resolve()
    (root/'source.csv').write_text('amount,unit\n12,USD\n')
    doc = read_document(root,'source.csv')
    extraction = extract_quantity(doc, QuantitySpec('cash',unit='USD',shape=(),axes=(),
                                  method='literal',time_basis='static'),
                                  selectors={'value':'amount','unit':'unit'})
    assert extraction['status'] == 'extracted' and extraction['candidates'][0]['value'] == 12
    def stage_compute(inputs,parameters,dependencies):
        return inputs['x']*2
    def stage_valid(value):
        return value == 6
    graph = Workflow([Stage('double',stage_compute,stage_valid,'1',inputs={'x':3},semantic_spec=spec)],
                     cache=StageCache(root/'cache',namespace='semantic-workflow'))
    assert graph.run()['results']['double'] == 6
    assert graph.run()['stages']['double']['cache'] == 'hit'
print("runtime-ok")
'''

WORKER_LIBRARY_SMOKE = r'''
import numpy as np
import pandas as pd
import qfkit
assert np.array_equal(qfkit.largest_remainder(7, [1.,1.,1.], tie_order=[30,10,20]), [2,3,2])
assert qfkit.mc_estimate([1.,3.,3.,1.], pair_antithetic=True)["se"] == 0.
rows = qfkit.numeric_time_frame(pd.DataFrame({"day":["2024-01-02","2024-01-01"],"x":["3","1"]}),
    date_col="day",value_cols=["x"],numeric_errors="raise",missing="raise")
assert rows["x"].tolist() == [1,3]
assert rows.index.tolist() == [pd.Timestamp("2024-01-01"),pd.Timestamp("2024-01-02")]
try:
    qfkit.numeric_time_frame(pd.DataFrame({"day":["2024-01-01"],"x":["bad"]}),
        date_col="day",value_cols=["x"],numeric_errors="raise",missing="raise")
except ValueError:
    pass
else:
    raise AssertionError("Expected strict numeric rejection")
''' + LABELLED_PANEL_SMOKE + STREAMING_SMOKE + CALENDAR_INPUT_SMOKE


# Both installed packages are required here; keep the single-package smoke scripts independent.
PANEL_BINDING_SMOKE = r'''
import hashlib
import json
import struct
import numpy as np
import pandas as pd
import qfkit
from qfruntime import (parse_panel_plan, encode_panel_label, decode_panel_label,
                       panel_artifact, load_panel_artifact, require_same_panel_axes)
pb_task = "Use early then late, integer 1 then string 1, features w then v; preserve missing cells."
pb_plan = {"schema":"panel-input-plan-v1", "op":"labelled_panel", "input":"rows.csv",
           "time_col":"t", "entity_col":"id", "value_cols":["w","v"],
           "times":[encode_panel_label(x) for x in ("early","late")],
           "entities":[encode_panel_label(x) for x in (1,"1")],
           "numeric_errors":"raise", "missing":"nan", "bind":"panel_input",
           "refs":[{"source_id":"instruction", "start":0, "end":len(pb_task), "quote":pb_task}]}
assert type(decode_panel_label(pb_plan["entities"][0])) is int
assert type(decode_panel_label(pb_plan["entities"][1])) is str
pb_binding = parse_panel_plan(json.dumps(pb_plan), columns_by_input={"rows.csv":["t","id","v","w"]},
    source_text=pb_task, source_sha256=hashlib.sha256(b"synthetic source assertion").hexdigest(),
    semantic_sha256=hashlib.sha256(b"synthetic semantic assertion").hexdigest())
pb_raw = pd.DataFrame({"t":["late","early","early"], "id":[1,"1",1],
                       "v":[21.,12.,11.], "w":[121.,np.nan,111.]}, index=[9,9,-2])
pb_before = pb_raw.copy(deep=True)
pb_text, pb_payload = panel_artifact(pb_raw, pb_binding)
pb_hash = hashlib.sha256(pb_text.encode()).hexdigest()
pb_panel, pb_receipt = load_panel_artifact(pb_text, pb_payload, expected_binding=pb_binding,
                                         expected_manifest_sha256=pb_hash)
np.testing.assert_array_equal(pb_panel["values"],
    [[[111.,11.],[np.nan,12.]],[[121.,21.],[np.nan,np.nan]]])
np.testing.assert_array_equal(pb_panel["source_positions"], [[2,1],[0,-1]])
np.testing.assert_array_equal(pb_panel["finite_mask"],
    [[[True,True],[False,True]],[[True,True],[False,False]]])
assert pb_panel["time_labels"] == ("early","late") and pb_panel["feature_labels"] == ("w","v")
assert pb_panel["entity_labels"] == (1,"1")
assert pb_receipt["source_dataframe_link"] == "caller_asserted"
assert pb_receipt["semantic_verification"] == "unverified"
pd.testing.assert_frame_equal(pb_raw, pb_before)
assert require_same_panel_axes(pb_panel, pb_panel) is None
def pb_reject(label, operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError(label)
pb_reject("missing cells", lambda: panel_artifact(pb_raw,
    dict(pb_binding, plan=dict(pb_plan, missing="raise"))))
pb_reject("axis drift", lambda: require_same_panel_axes(pb_panel,
    dict(pb_panel, entity_labels=("1",1))))
pb_reject("manifest hash drift", lambda: load_panel_artifact(pb_text, pb_payload,
    expected_binding=pb_binding, expected_manifest_sha256="0"*64))
pb_reject("payload hash drift", lambda: load_panel_artifact(pb_text, pb_payload[:-1]+b"\x01",
    expected_binding=pb_binding, expected_manifest_sha256=pb_hash))
# Rehash a malformed position to reach structural validation, not just the byte hash gate.
pb_bad_payload = bytearray(pb_payload)
struct.pack_into("<q", pb_bad_payload, 2*2*2*8, 99)
pb_bad_payload = bytes(pb_bad_payload)
pb_bad_manifest = json.loads(pb_text)
pb_bad_manifest["payload_sha256"] = hashlib.sha256(pb_bad_payload).hexdigest()
pb_bad_text = json.dumps(pb_bad_manifest, allow_nan=False)
pb_reject("source position drift", lambda: load_panel_artifact(pb_bad_text, pb_bad_payload,
    expected_binding=pb_binding, expected_manifest_sha256=hashlib.sha256(pb_bad_text.encode()).hexdigest()))
'''


IMPORT_SMOKE = ("WORKER_LIBRARY_SMOKE = " + repr(WORKER_LIBRARY_SMOKE)
                + "\nRUNTIME_SMOKE = " + repr(RUNTIME_SMOKE)
                + "\nPANEL_BINDING_SMOKE = " + repr(PANEL_BINDING_SMOKE) + "\n") + r'''
import arch, backtrader, numba, openpyxl, plotly, polars, requests, seaborn, sklearn, statsmodels, talib, tqdm
import pandas as pd
frame = pd.DataFrame({"a": [1.5, 2.5]})
frame.to_excel("/tmp/smoke.xlsx", index=False)
assert pd.read_excel("/tmp/smoke.xlsx").equals(frame)
import pathlib, subprocess, sys, tempfile
exec(RUNTIME_SMOKE)
exec(PANEL_BINDING_SMOKE)
''' + LIBRARY_SMOKE + r'''
from t1_phase1.baseline import library_catalog, library86_on, s12_lib_on, worker_environment, worker_site
# Explicit profiles: legacy 0.4, S12 0.5, full installed 0.6.
limit = qfkit.CATALOG_050_LEN if s12_lib_on() else qfkit.CATALOG_040_LEN
assert library_catalog() == (qfkit.CATALOG if library86_on() else qfkit.CATALOG[:limit])
root = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
site = root / "sandbox"
site.mkdir()
(site / "sitecustomize.py").write_text(worker_site())
script = root / "generated.py"
script.write_text("import qfkit\nprint(round(qfkit.bsm_price(100, 100, 1, 0.05, 0.2, kind='call'), 4))\n" + RUNTIME_SMOKE + WORKER_LIBRARY_SMOKE + PANEL_BINDING_SMOKE)
done = subprocess.run([sys.executable, str(script)], env=worker_environment(site=site), cwd=root,
                      capture_output=True, text=True, timeout=120)
assert done.returncode == 0 and done.stdout.splitlines() == ["10.4506", "runtime-ok"], (done.returncode, done.stderr[-2000:])
print("imports-ok")
'''


def import_smoke(context, image_id):
    """Import the analysis packages offline; xlsx inputs need a pandas Excel engine. The optional library
    qfkit must import and compute (a known value from each module) both here and in a program run the way the
    worker runs one (its own environment, without PYTHONPATH=/opt, under the sandbox sitecustomize), and the
    controller must find the catalog it shows the model."""
    command = ['docker', '--context', context, 'run', '--rm', '--network=none', '--read-only',
               '--user=65534:65534', '--tmpfs=/tmp:rw,nosuid,size=64m', image_id,
               'python', '-c', IMPORT_SMOKE]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'returncode': None, 'error': type(exc).__name__}
    return {'ok': completed.returncode == 0 and 'imports-ok' in (completed.stdout or ''),
            'returncode': completed.returncode}


CRASH_SMOKE = r'''
import pathlib, subprocess, sys, tempfile
from t1_phase1.baseline import worker_environment, worker_site
root = pathlib.Path(tempfile.mkdtemp(dir="/tmp"))
site = root / "sandbox"
site.mkdir()
(site / "sitecustomize.py").write_text(worker_site())
script = root / "generated.py"
script.write_text("import numpy as np\nimport pandas as pd\n\n\ndef weights(frame):\n"
                  "    raw = np.array([1, 'a'], dtype=object)\n    return np.isfinite(raw)\n\n\n"
                  "weights(pd.DataFrame({'date': ['2024-01-02'], 'px': [1.5]}))\n")
report = root / "crash_report.txt"
env = dict(worker_environment(site=site), T1_CRASH_REPORT=str(report), T1_PROGRAM=str(script))
done = subprocess.run([sys.executable, str(script)], env=env, cwd=root, capture_output=True, timeout=120)
text = report.read_text()
assert done.returncode == 1, done.returncode
for part in ("TypeError: ufunc 'isfinite' not supported", "in weights: return np.isfinite(raw)",
             "raw = ndarray(dtype=object, shape=(2,)) first [1, 'a']",
             "frame = DataFrame(shape=(1, 2)) columns ['date':object, 'px':float64] index RangeIndex from 0"):
    assert part in text, (part, text)
print("crash-report-ok")
'''


def crash_smoke(context, image_id):
    """The worker's crash hook describes numpy and pandas values in the built image, offline."""
    command = ['docker', '--context', context, 'run', '--rm', '--network=none', '--read-only',
               '--user=65534:65534', '--tmpfs=/tmp:rw,noexec,nosuid,size=64m', image_id,
               'python', '-c', CRASH_SMOKE]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'ok': False, 'returncode': None, 'error': type(exc).__name__}
    return {'ok': completed.returncode == 0 and 'crash-report-ok' in (completed.stdout or ''),
            'returncode': completed.returncode}


PACKAGES = ('t1_phase1', 'qfkit/qfkit', 'qfruntime/qfruntime')  # controller and optional source-only helpers


def source_files(root=ROOT):
    """{context path: source} for the image's own files: the pins, the solve verb and each package's .py
    modules. Tests, caches and task files never enter the context."""
    files = {'requirements-container.txt': root / 'requirements-container.txt',
             'scripts/solve': root / 'scripts/solve'}
    for package in PACKAGES:
        for path in sorted((root / package).glob('*.py')):
            files[f'{package}/{path.name}'] = path
    return files


def execute(args, out, report):
    manifest_path = HERE / 'wheelhouse-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'verified' or manifest.get('python') != '3.13'
            or manifest.get('platform') != 'linux/x86_64'):
        raise ValueError('Wheelhouse must have a completed verification receipt for this target')
    metadata_path = HERE / 'wheel-metadata.json'
    if digest(metadata_path) != manifest['metadata_sha256']:
        raise ValueError('Official wheel metadata hash mismatch')
    metadata = json.loads(metadata_path.read_text())
    fields = ('name', 'version', 'filename', 'url', 'size', 'sha256')
    if [{key: item[key] for key in fields} for item in manifest['wheels']] != metadata['files']:
        raise ValueError('Wheel manifest differs from official metadata')
    lock = HERE / 'requirements-linux-amd64.lock'
    if digest(lock) != manifest['lock_sha256']:
        raise ValueError('Lockfile hash mismatch')
    locked = pins(lock)
    if manifest['file_count'] != len(manifest['wheels']) or len(manifest['wheels']) != len(locked):
        raise ValueError('Wheel count differs from the complete lockfile')
    direct = pins(ROOT / 'requirements-container.txt')
    if any(locked.get(name) != version for name, version in direct.items()):
        raise ValueError('Direct dependency pin mismatch')
    controller_sources = source_files()
    provenance = build_provenance({**controller_sources, 'Dockerfile': ROOT/'Dockerfile.amd64'})
    verified_sources = provenance.pop('verified_source_sha256')
    report['controller_source'] = provenance
    if provenance['source_state'] != 'frozen':
        raise ValueError('Controller build inputs are not frozen at a known source commit')
    files = {'requirements-offline.lock': lock, **controller_sources}
    wheel_versions = {}
    for item in manifest['wheels']:
        name = item['filename']
        if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl', name):
            raise ValueError('Unsafe wheel filename')
        path = HERE / 'wheels' / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != item['size']
                or digest(path) != item['sha256']):
            raise ValueError('Wheel size/hash verification failed')
        package = re.sub(r'[-_.]+', '-', item['name'].lower())
        if package in wheel_versions:
            raise ValueError('Duplicate wheel package')
        wheel_versions[package] = item['version']
        files['wheels/' + name] = path
    if wheel_versions != locked:
        raise ValueError('Wheel roster differs from the complete lockfile')
    selected = getattr(args, 'solver_profile', profiles.CANDIDATE)
    original_recipe = read_bound_file(ROOT/'Dockerfile.amd64', verified_sources['Dockerfile'])
    stale = recipe_default_problems(original_recipe.decode())
    if stale:
        raise ValueError('Dockerfile.amd64 does not carry the image default of ' + ', '.join(stale)
                         + ' (scripts/thin_defaults.py sets solver_profile and both recipes)')
    dockerfile = profile_recipe(original_recipe.decode(), selected)
    dockerfile += '\nLABEL org.opencontainers.image.revision="'+provenance['source_commit']+'"\n'
    recipe = out / 'Dockerfile.offline'
    recipe.write_text(dockerfile)
    files['Dockerfile'] = recipe
    hashes = {**{name:verified_sources[name] for name in controller_sources},
              'requirements-offline.lock': manifest['lock_sha256'],
              **{'wheels/'+item['filename']:item['sha256'] for item in manifest['wheels']},
              'Dockerfile':hashlib.sha256(dockerfile.encode()).hexdigest()}
    provenance['context_sha256'] = context_binding(hashes)
    identity_path = out/'build_identity.py'
    identity_path.write_text('"""Generated frozen build provenance; no credentials."""\nIDENTITY = '+repr(provenance)+'\n')
    files['t1_phase1/build_identity.py'] = identity_path
    hashes['t1_phase1/build_identity.py'] = digest(identity_path)
    (out / 'context-manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    report.update(solver_profile=selected, expected_solver_env=profiles.profile_env(selected),
                  context_binding_excludes=['t1_phase1/build_identity.py'],
                  context_manifest_sha256=digest(out/'context-manifest.json'),
                  verified_wheels=len(wheel_versions), direct_pins_unchanged=True,
                  wheel_manifest_sha256=digest(manifest_path), lock_sha256=digest(lock),
                  source_dockerfile_sha256=verified_sources['Dockerfile'],
                  actual_dockerfile_sha256=digest(recipe), build_run_network='none')
    with tempfile.TemporaryFile(dir=out) as context:
        with tarfile.open(fileobj=context, mode='w') as archive:
            for name, path in sorted(files.items()):
                add_bound_file(archive, name, path, hashes[name])
        context.seek(0)
        command = ['docker', '--context', args.context, 'build', '--network=none',
                   '--platform', 'linux/amd64', '--pull=false', '-t', args.image, '-f', 'Dockerfile', '-']
        with (out / 'build.log').open('w') as log:
            completed = subprocess.run(command, stdin=context, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=600, check=False)
        report['build_returncode'] = completed.returncode
        if completed.returncode:
            raise RuntimeError('Offline image build failed; see build.log')
    post_build_checks(args.context, args.image, report)
    report['status'] = 'built_not_integration_tested'


def post_build_checks(context, image_tag, report):
    """Refuse the build unless the image meets the contract and runs offline by image ID."""
    completed = subprocess.run(['docker', '--context', context, 'image', 'inspect', image_tag],
                               capture_output=True, text=True, timeout=20, check=True)
    image = json.loads(completed.stdout)[0]
    report['image'] = {key: image.get(key) for key in ('Id', 'Architecture', 'Os')}
    problems = contract_problems(image, solver_profile=report.get('solver_profile'))
    report['image_contract_problems'] = problems
    if problems:
        raise ValueError('Built image violates the image contract: ' + ', '.join(problems))
    report['solve_smoke'] = solve_smoke(context, image['Id'])
    if not report['solve_smoke']['ok']:
        raise ValueError('The solve verb did not run from PATH in the built image')
    report['import_smoke'] = import_smoke(context, image['Id'])
    if not report['import_smoke']['ok']:
        raise ValueError('Analysis packages did not import in the built image')
    report['crash_smoke'] = crash_smoke(context, image['Id'])
    if not report['crash_smoke']['ok']:
        raise ValueError('The crash-report hook did not work in the built image')
    if report.get('solver_profile') is not None:
        report['identity_smoke'] = identity_smoke(context, image['Id'], report['solver_profile'],
                                                report['controller_source'])
        if not report['identity_smoke']['ok']:
            raise ValueError('Built solver identity differs from the frozen profile or source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--context', default='colima')
    parser.add_argument('--image', default='t1-phase1:b0-amd64')
    parser.add_argument('--solver-profile', choices=(profiles.CANDIDATE,),
                        default=profiles.CANDIDATE)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'not_started', 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'context': args.context, 'image_tag': args.image, 'official_house_calls': 0}
    try:
        execute(args, out, report)
    except Exception as exc:
        report.update(status='build_error', reason=type(exc).__name__ + ': ' + str(exc))
    report['duration_sec'] = round(time.monotonic() - started, 3)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if report['status'] == 'built_not_integration_tested' else 1


if __name__ == '__main__':
    raise SystemExit(main())
