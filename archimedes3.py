import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
theta = pd.read_csv("DeltaAil_subfile_2.csv", skiprows=19000, skipfooter=9000).squeeze().to_numpy()
I     = pd.read_csv("IservoAil_subfile_2.csv", skiprows=19000, skipfooter=9000).squeeze().to_numpy()

dt = 0.00001
t  = np.arange(len(theta)) * dt

# ── 2. ROBUST SMOOTHING + DERIVATIVES ─────────────────────────────────────────
# First: mild smoothing to suppress spikes
theta_pre = savgol_filter(theta, 51, 3)

# Then: local polynomial differentiation (piecewise!)
window = int(0.00004 / dt)  
if window % 2 == 0:
    window += 1

order = 4

theta_smooth = savgol_filter(theta_pre, window, order)
dtheta       = savgol_filter(theta_pre, window, order, deriv=1, delta=dt)
d2theta      = savgol_filter(theta_pre, window, order, deriv=2, delta=dt)

# ── 3. BUILD FEATURE MATRIX ───────────────────────────────────────────────────
X = np.column_stack([I, dtheta, theta_smooth])
y = d2theta

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ── 5. NORMALIZE (IMPORTANT!) ─────────────────────────────────────────────────
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)

X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std

# ── 6. SOLVE USING LEAST SQUARES (NO GD!) ─────────────────────────────────────
coeffs_n = np.linalg.lstsq(X_train_n, y_train, rcond=None)[0]

# Convert back to original scale
coeffs = coeffs_n / X_std

# ── 7. EXTRACT RATIOS ─────────────────────────────────────────────────────────
BoverJ =  coeffs[0]
KoverJ = -coeffs[1]
AoverJ = -coeffs[2]

# ── 8. EVALUATION ─────────────────────────────────────────────────────────────
y_test_pred = X_test @ coeffs

ss_res = np.sum((y_test - y_test_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2     = 1 - ss_res / ss_tot

print("\n" + "=" * 30)
print(f"{'Ratio':<8} {'Value':>12}")
print("-" * 30)
for name, val in [("B/J", BoverJ), ("K/J", KoverJ), ("A/J", AoverJ)]:
    print(f"{name:<8} {val:>12.4f}")
print("=" * 30)
print(f"R² on test set: {r2:.6f}")

# ── 9. SIMULATE ODE ───────────────────────────────────────────────────────────
def ode_est(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = BoverJ * I_val - KoverJ * dth - AoverJ * th
    return [dth, d2th]

# Better initial conditions (avoid edge noise)
theta0  = np.mean(theta[:10])
dtheta0 = np.mean(dtheta[:10])

sol_est = solve_ivp(
    ode_est,
    [t[0], t[-1]],
    [theta0, dtheta0],
    t_eval=t,
    max_step=dt
)

theta_est = sol_est.y[0]

# ── 10. PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(10, 10))

fig.suptitle("Robust ODE Identification (Erratic Data)", fontsize=13)

# θ comparison
axes[0].plot(t, theta, label="Measured θ(t)", alpha=0.4)
axes[0].plot(t, theta_est, '--', label="Estimated θ(t)")
axes[0].set_ylabel("θ (rad)")
axes[0].legend()
axes[0].grid(True)

# Input
axes[1].plot(t, I, label="I(t)")
axes[1].set_ylabel("Current")
axes[1].legend()
axes[1].grid(True)

# Derivatives (sanity check!)
axes[2].plot(t, dtheta, label="θ̇")
axes[2].plot(t, d2theta, label="θ̈")
axes[2].set_title("Estimated derivatives")
axes[2].legend()
axes[2].grid(True)

# Prediction vs truth (test set)
axes[3].plot(y_test, label="True θ̈", alpha=0.5)
axes[3].plot(y_test_pred, label="Predicted θ̈", linestyle="--")
axes[3].set_title("Test set fit")
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.savefig("ode_plot_improved.png", dpi=150)
plt.show()