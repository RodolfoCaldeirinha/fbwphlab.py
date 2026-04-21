import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
theta = pd.read_csv("DeltaAil_subfile_2.csv",
                    skiprows=9000, skipfooter=9000,
                    engine="python").squeeze().to_numpy()

I = pd.read_csv("IservoAil_subfile_2.csv",
                skiprows=9000, skipfooter=9000,
                engine="python").squeeze().to_numpy()

dt = 0.0001
t = np.arange(len(theta)) * dt

# ── 2. DOWNSAMPLE (IMPORTANT FOR SPEED + STABILITY) ───────────────────────────
step = 100  # adjust if needed
t = t[::step]
theta = theta[::step]
I = I[::step]

dt = t[1] - t[0]

# -- Normalize for better optimization convergence --------------------------------

theta_mean = np.mean(theta)
theta_std  = np.std(theta)

I_mean = np.mean(I)
I_std  = np.std(I)

theta_n = (theta - theta_mean) / theta_std
I_n     = (I - I_mean) / I_std

# ── 3. SIMULATION FUNCTION ────────────────────────────────────────────────────
def simulate_theta(params):
    B, K, A, dtheta0 = params

    def ode(t_val, y):
        th, dth = y
        I_val = np.interp(t_val, t, I_n)
        d2th = B * I_val - K * dth - A * th
        return [dth, d2th]

    sol = solve_ivp(
        ode,
        [t[0], t[-1]],
        [theta_n[0], dtheta0],
        t_eval=t,
        max_step=dt
    )

    return sol.y[0]

# ── 4. RESIDUAL FUNCTION (WITH REGULARIZATION) ────────────────────────────────
lambda_reg = 0.0  # tune this (0.001 → 0.1)

def residuals(params):
    theta_sim = simulate_theta(params)

    # Data fit error
    data_error = theta_sim - theta

    # Regularization (penalize large parameters)
    reg = lambda_reg * np.array(params)

    return np.concatenate([data_error, reg])


# ── 5. INITIAL GUESS ──────────────────────────────────────────────────────────
params0 = [0.03, 0.6, 6.2, -0.05]  # [B/J, K/J, A/J, initial velocity]

# ── 6. OPTIMIZATION ───────────────────────────────────────────────────────────
result = least_squares(residuals, params0, verbose=2)

BoverJ, KoverJ, AoverJ, dtheta0 = result.x

# ── 7. FINAL SIMULATION ───────────────────────────────────────────────────────
theta_est = simulate_theta(result.x)

# ── 8. GOODNESS OF FIT ────────────────────────────────────────────────────────
ss_res = np.sum((theta - theta_est) ** 2)
ss_tot = np.sum((theta - np.mean(theta)) ** 2)
r2 = 1 - ss_res / ss_tot

# ── 9. PRINT RESULTS ──────────────────────────────────────────────────────────
print("\n" + "=" * 35)
print(f"{'Parameter':<10} {'Value':>12}")
print("-" * 35)
print(f"{'B/J':<10} {BoverJ:>12.6f}")
print(f"{'K/J':<10} {KoverJ:>12.6f}")
print(f"{'A/J':<10} {AoverJ:>12.6f}")
print(f"{'θ̇₀':<10} {dtheta0:>12.6f}")
print("=" * 35)
print(f"R² (θ fit): {r2:.6f}")

# ── 10. PLOT ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(t, theta, label="Measured θ(t)", alpha=0.5)
plt.plot(t, theta_est, '--', label="Estimated θ(t)")
plt.xlabel("Time (s)")
plt.ylabel("θ (rad)")
plt.title("Direct θ Fitting (ODE Identification)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("theta_fit_direct.png", dpi=150)
plt.show()