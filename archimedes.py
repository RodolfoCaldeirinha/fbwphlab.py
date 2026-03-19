"""
Parameter identification for: J*theta_ddot + k*theta_dot + A*theta = B*I

Strategy:
  1. Load your measured time, theta, and I data
  2. Use scipy.optimize.minimize to iteratively simulate forward with
     candidate [J, k, A, B] and minimize the sum-of-squared error
     against your measured theta.
  3. Multiple random restarts avoid local minima.
  4. Reports confidence via residuals and parameter sensitivity.

Requirements:
    pip install numpy scipy matplotlib pandas
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt


# =============================================================================
# 1. LOAD YOUR DATA
#    Replace this section with your actual data loading.
#    Expected: 1D arrays of equal length for t, theta, I_input
# =============================================================================

def load_data():
    """
    Replace this with your real data.
    Must return:
        t       : 1D array, time points (s), uniformly or non-uniformly spaced
        theta   : 1D array, measured angle at each time point
        I_input : 1D array, input current/torque at each time point
    """
    # --- EXAMPLE: synthetic data with known parameters ---
    J_true, k_true, A_true, B_true = 2.0, 0.8, 3.0, 1.5
    t = np.linspace(0, 10, 1000)
    I_input = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 1.2 * t)

    def ode(t_val, y, I_interp):
        theta, theta_dot = y
        I_val = float(I_interp(t_val))
        theta_ddot = (B_true * I_val - k_true * theta_dot - A_true * theta) / J_true
        return [theta_dot, theta_ddot]

    from scipy.interpolate import interp1d
    I_func = interp1d(t, I_input, kind='linear', fill_value='extrapolate')
    sol = solve_ivp(ode, [t[0], t[-1]], [0.1, 0.0], t_eval=t,
                    args=(I_func,), method='RK45', rtol=1e-8, atol=1e-10)
    theta = sol.y[0] + np.random.normal(0, 0.002, len(t))  # small noise
    print(f"Synthetic data generated. True params: J={J_true}, k={k_true}, A={A_true}, B={B_true}")
    return t, theta, I_input


# =============================================================================
# 2. FORWARD SIMULATION
#    Given candidate params, simulate the ODE and return predicted theta.
# =============================================================================

def simulate(params, t, I_input, theta0=None, theta_dot0=0.0):
    """
    Simulate J*theta_ddot + k*theta_dot + A*theta = B*I
    and return predicted theta at the same time points as t.

    params  : [J, k, A, B]
    t       : time array
    I_input : input array (same length as t)
    theta0  : initial theta (defaults to first measured value)
    """
    J, k, A, B = params

    # Basic sanity: skip clearly unphysical parameter sets
    if J <= 0 or k < 0 or A < 0 or B <= 0:
        return None

    from scipy.interpolate import interp1d
    I_func = interp1d(t, I_input, kind='linear', fill_value='extrapolate')

    def ode(t_val, y):
        th, thd = y
        I_val = float(I_func(t_val))
        th_ddot = (B * I_val - k * thd - A * th) / J
        return [thd, th_ddot]

    y0 = [theta0 if theta0 is not None else 0.0, theta_dot0]

    try:
        sol = solve_ivp(
            ode, [t[0], t[-1]], y0,
            t_eval=t, method='RK45',
            rtol=1e-6, atol=1e-8,
            max_step=(t[-1] - t[0]) / 200
        )
        if not sol.success or np.any(np.isnan(sol.y[0])):
            return None
        return sol.y[0]
    except Exception:
        return None


# =============================================================================
# 3. COST FUNCTION
#    Sum of squared errors between simulated and measured theta.
#    Uses log(J) etc. internally to enforce positivity without bounds.
# =============================================================================

def cost(log_params, t, theta_meas, I_input, theta0):
    """Cost in log-parameter space to enforce J,k,A,B > 0 naturally."""
    params = np.exp(log_params)
    theta_pred = simulate(params, t, I_input, theta0=theta0)
    if theta_pred is None:
        return 1e12
    residuals = theta_pred - theta_meas
    return np.sum(residuals ** 2)


# =============================================================================
# 4. IDENTIFY PARAMETERS
#    Two-stage approach:
#      Stage 1: global search with differential evolution (escapes local minima)
#      Stage 2: local refinement with Nelder-Mead / L-BFGS-B
# =============================================================================

def identify_parameters(t, theta_meas, I_input,
                         initial_guess=None,
                         param_bounds=None,
                         n_restarts=10,
                         verbose=True):
    """
    Find [J, k, A, B] that best fits the measured theta.

    Parameters
    ----------
    t             : time array
    theta_meas    : measured theta array
    I_input       : input array
    initial_guess : [J, k, A, B] starting point (optional)
    param_bounds  : [(J_min,J_max), (k_min,k_max), (A_min,A_max), (B_min,B_max)]
                    Defaults to broad search if None
    n_restarts    : number of random restarts for local optimizer
    verbose       : print progress

    Returns
    -------
    dict with best params, error, and all restart results
    """
    theta0 = theta_meas[0]

    if param_bounds is None:
        # Broad default bounds — tighten these if you have domain knowledge!
        param_bounds = [(0.01, 50), (0.001, 20), (0.001, 50), (0.01, 20)]

    # Log-space bounds for optimizer
    log_bounds = [(np.log(lo), np.log(hi)) for (lo, hi) in param_bounds]

    # ---- Stage 1: Global search (differential evolution) ----
    if verbose:
        print("\n--- Stage 1: Global search (differential evolution) ---")

    de_result = differential_evolution(
        cost, log_bounds,
        args=(t, theta_meas, I_input, theta0),
        maxiter=500, tol=1e-8, seed=42,
        popsize=15, mutation=(0.5, 1.5), recombination=0.7,
        disp=verbose, workers=1
    )
    best_log_params = de_result.x
    best_cost = de_result.fun
    if verbose:
        p = np.exp(best_log_params)
        print(f"Global best: J={p[0]:.4f}, k={p[1]:.4f}, A={p[2]:.4f}, B={p[3]:.4f}  | SSE={best_cost:.6f}")

    # ---- Stage 2: Local refinement from best + random restarts ----
    if verbose:
        print(f"\n--- Stage 2: Local refinement ({n_restarts} restarts) ---")

    all_results = []

    # Always include DE result and user initial guess as starting points
    starts = [best_log_params]
    if initial_guess is not None:
        starts.append(np.log(np.array(initial_guess, dtype=float)))

    rng = np.random.default_rng(0)
    for _ in range(n_restarts - len(starts)):
        rand_start = np.array([rng.uniform(lo, hi) for (lo, hi) in log_bounds])
        starts.append(rand_start)

    for i, start in enumerate(starts):
        res = minimize(
            cost, start,
            args=(t, theta_meas, I_input, theta0),
            method='L-BFGS-B',
            bounds=log_bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        p = np.exp(res.x)
        all_results.append({'params': p, 'cost': res.fun, 'success': res.success})
        if res.fun < best_cost:
            best_cost = res.fun
            best_log_params = res.x
        if verbose:
            print(f"  Restart {i+1:2d}: J={p[0]:.4f}, k={p[1]:.4f}, A={p[2]:.4f}, B={p[3]:.4f}  | SSE={res.fun:.6f}")

    best_params = np.exp(best_log_params)

    # ---- Final polish with Nelder-Mead (no gradient needed) ----
    res_nm = minimize(
        cost, best_log_params,
        args=(t, theta_meas, I_input, theta0),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-14}
    )
    if res_nm.fun < best_cost:
        best_log_params = res_nm.x
        best_params = np.exp(best_log_params)
        best_cost = res_nm.fun

    # ---- RMSE ----
    theta_pred = simulate(best_params, t, I_input, theta0=theta0)
    rmse = np.sqrt(np.mean((theta_pred - theta_meas) ** 2))
    rel_rmse = rmse / (np.max(theta_meas) - np.min(theta_meas)) * 100

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Best fit parameters:")
        print(f"    J = {best_params[0]:.6f}")
        print(f"    k = {best_params[1]:.6f}")
        print(f"    A = {best_params[2]:.6f}")
        print(f"    B = {best_params[3]:.6f}")
        print(f"  RMSE          : {rmse:.6f}")
        print(f"  Relative RMSE : {rel_rmse:.2f}%")
        print(f"{'='*50}\n")

    return {
        'params': best_params,
        'J': best_params[0], 'k': best_params[1],
        'A': best_params[2], 'B': best_params[3],
        'rmse': rmse,
        'relative_rmse_pct': rel_rmse,
        'best_cost': best_cost,
        'all_restarts': all_results
    }


# =============================================================================
# 5. SENSITIVITY ANALYSIS
#    Perturb each parameter ±5% and see how much the cost changes.
#    High sensitivity = well-identified parameter.
#    Low sensitivity = parameter may be underdetermined by your data.
# =============================================================================

def sensitivity_analysis(best_params, t, theta_meas, I_input, perturb=0.05):
    """Check how sensitive the cost is to each parameter (±perturb fraction)."""
    theta0 = theta_meas[0]
    log_p = np.log(best_params)
    base_cost = cost(log_p, t, theta_meas, I_input, theta0)

    labels = ['J', 'k', 'A', 'B']
    sensitivities = {}
    print("Sensitivity analysis (cost change for ±5% parameter perturbation):")
    print(f"  {'Param':<6} {'Cost at -5%':>14} {'Cost at base':>14} {'Cost at +5%':>14} {'Sensitivity':>14}")
    for i, name in enumerate(labels):
        p_lo = log_p.copy(); p_lo[i] -= perturb
        p_hi = log_p.copy(); p_hi[i] += perturb
        c_lo = cost(p_lo, t, theta_meas, I_input, theta0)
        c_hi = cost(p_hi, t, theta_meas, I_input, theta0)
        sens = (c_lo + c_hi - 2*base_cost) / (perturb**2)
        sensitivities[name] = sens
        print(f"  {name:<6} {c_lo:>14.4f} {base_cost:>14.4f} {c_hi:>14.4f} {sens:>14.2f}")
    return sensitivities


# =============================================================================
# 6. PLOT RESULTS
# =============================================================================

def plot_results(t, theta_meas, I_input, result):
    params = result['params']
    theta_pred = simulate(params, t, I_input, theta0=theta_meas[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle('Parameter Identification Results', fontsize=13)

    ax = axes[0]
    ax.plot(t, theta_meas, 'b-', lw=1, alpha=0.7, label='Measured θ')
    ax.plot(t, theta_pred, 'r--', lw=1.5, label='Simulated θ (identified params)')
    ax.set_ylabel('θ (rad)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = theta_pred - theta_meas
    ax.plot(t, residuals, 'g-', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Residual (rad)')
    ax.set_title(f'Residuals  |  RMSE={result["rmse"]:.5f}  |  Rel. RMSE={result["relative_rmse_pct"]:.2f}%', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t, I_input, 'k-', lw=1)
    ax.set_ylabel('I (input)')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

    param_text = (f"J = {result['J']:.5f}\n"
                  f"k = {result['k']:.5f}\n"
                  f"A = {result['A']:.5f}\n"
                  f"B = {result['B']:.5f}")
    axes[0].text(0.01, 0.97, param_text, transform=axes[0].transAxes,
                 fontsize=8, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('identification_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to identification_results.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # 1. Load data
    t, theta_meas, I_input = load_data()

    # 2. Run identification
    #    - Set param_bounds to realistic ranges for your system
    #    - Set initial_guess if you have a rough idea of the values
    result = identify_parameters(
        t, theta_meas, I_input,
        initial_guess=None,          # e.g. [2.0, 0.5, 3.0, 1.0]
        param_bounds=[               # tighten these with domain knowledge
            (0.1, 20),               # J
            (0.001, 10),             # k
            (0.1, 20),               # A
            (0.1, 10),               # B
        ],
        n_restarts=10,
        verbose=True
    )

    # 3. Sensitivity analysis
    sensitivity_analysis(result['params'], t, theta_meas, I_input)

    # 4. Plot
    plot_results(t, theta_meas, I_input, result)