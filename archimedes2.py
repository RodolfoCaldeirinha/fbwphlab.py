import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ── 1. Generate synthetic data ────────────────────────────────────────────────
# ODE:  J·θ̈  +  K·θ̇  +  A·θ  =  B·I(t)

TRUE = dict(J=2.0, K=0.8, A=1.5, B=1.2)

dt   = 0.01
t    = np.arange(0, 10, dt)
I    = np.sin(2 * t) + 0.5 * np.cos(5 * t)   # forcing current

def ode(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = (TRUE["B"] * I_val - TRUE["K"] * dth - TRUE["A"] * th) / TRUE["J"]
    return [dth, d2th]

sol   = solve_ivp(ode, [t[0], t[-1]], [0.0, 0.0], t_eval=t, max_step=dt)
theta = sol.y[0]

# Add a little noise so it feels like real data
theta += np.random.normal(0, 0.01, size=theta.shape)

# ── 2. Numerical derivatives (Savitzky-Golay — cleaner than np.gradient) ─────
theta   = savgol_filter(theta, window_length=11, polyorder=3)
dtheta  = savgol_filter(theta, window_length=11, polyorder=3, deriv=1, delta=dt)
d2theta = savgol_filter(theta, window_length=11, polyorder=3, deriv=2, delta=dt)

# ── 3. Build feature matrix ───────────────────────────────────────────────────
# Rearranging:  θ̈ = (B/J)·I  -  (K/J)·θ̇  -  (A/J)·θ
# So regress:   θ̈  ~  [I,  θ̇,  θ]   →  coefficients [B/J,  -K/J,  -A/J]

X = np.column_stack([I, dtheta, theta])   # features
y = d2theta                                # target

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False   # shuffle=False keeps time order
)

# ── 5. Fit linear regression ──────────────────────────────────────────────────
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

c1, c2, c3 = model.coef_   # c1 = B/J,  c2 = -K/J,  c3 = -A/J

# ── 6. Recover J, K, A, B  (need one anchor — fix J = true J for scale) ──────
# Without knowing J independently, the system is only identifiable up to scale.
# In practice you know J (e.g. from a mass measurement), so we fix it here.
J_known = TRUE["J"]

J_est = J_known
B_est =  c1 * J_est
K_est = -c2 * J_est
A_est = -c3 * J_est

# ── 7. Results ────────────────────────────────────────────────────────────────
r2 = model.score(X_test, y_test)

print("=" * 40)
print(f"{'Param':<8} {'True':>8} {'Estimated':>12} {'Error %':>10}")
print("-" * 40)
for name, true, est in [("J", TRUE["J"], J_est),
                         ("K", TRUE["K"], K_est),
                         ("A", TRUE["A"], A_est),
                         ("B", TRUE["B"], B_est)]:
    err = abs(est - true) / true * 100
    print(f"{name:<8} {true:>8.4f} {est:>12.4f} {err:>9.2f}%")
print("=" * 40)
print(f"R² on test set: {r2:.6f}")

import matplotlib.pyplot as plt
 
def ode_est(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = (B_est * I_val - K_est * dth - A_est * th) / J_est
    return [dth, d2th]
 
sol_est   = solve_ivp(ode_est, [t[0], t[-1]], [0.0, 0.0], t_eval=t, max_step=dt)
theta_est = sol_est.y[0]
 
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("ODE: true vs estimated coefficients", fontsize=13)
 
axes[0].plot(t, theta,     label="True θ(t)",      color="steelblue", linewidth=2)
axes[0].plot(t, theta_est, label="Estimated θ(t)", color="tomato",    linewidth=1.5, linestyle="--")
axes[0].set_ylabel("θ  (rad)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
 
axes[1].plot(t, I, label="I(t) forcing", color="seagreen", linewidth=1.5)
axes[1].set_ylabel("Current  I(t)")
axes[1].set_xlabel("Time  (s)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("ode_plot.png", dpi=150)
plt.show()
print("Plot saved to ode_plot.png")