import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── 1. Generate synthetic data ────────────────────────────────────────────────
# ODE:  J·θ̈  +  K·θ̇  +  A·θ  =  B·I(t)

def convert_array(file):
    with open(file, 'r') as f:
        lines = [line.rstrip() for line in f]
    array = []
    for line in lines:
        array.append(line)
    array.pop(0)
    array = [float(x) for x in array]
    return array

theta = convert_array("Theta_subfile_1.csv")

TRUE = dict(J=5.0, K=0.3, A=8, B=1.2)

#dt  = 0.05
#t   = np.arange(0, 10, dt)
#I   = np.sin(2 * t) + 0.5 * np.cos(5 * t)   # forcing current



def ode(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = (TRUE["B"] * I_val - TRUE["K"] * dth - TRUE["A"] * th) / TRUE["J"]
    return [dth, d2th]

sol   = solve_ivp(ode, [t[0], t[-1]], [0.0, 0.0], t_eval=t, max_step=dt)
#theta = sol.y[0] + np.random.normal(0, 0.01, sol.y[0].shape)

# ── 2. Numerical derivatives ──────────────────────────────────────────────────
theta   = savgol_filter(theta, window_length=11, polyorder=3)
dtheta  = savgol_filter(theta, window_length=11, polyorder=3, deriv=1, delta=dt)
d2theta = savgol_filter(theta, window_length=11, polyorder=3, deriv=2, delta=dt)

# ── 3. Build feature matrix ───────────────────────────────────────────────────
# Rearranging:  θ̈ = (B/J)·I  -  (K/J)·θ̇  -  (A/J)·θ
# So regress:   θ̈  ~  [I,  θ̇,  θ]   →  coefficients [B/J,  -K/J,  -A/J]

X = np.column_stack([I, dtheta, theta])
y = d2theta

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False    # shuffle=False keeps time order
)

# ── 5. Gradient descent ───────────────────────────────────────────────────────
# lr is small because features are unscaled — this lets us see the loss
# curve actually descend across many epochs rather than converging in 1 step.
lr     = 0.01
epochs = 50_000
coeffs = np.zeros(3)
n      = len(X_train)
losses = []

for epoch in range(epochs):
    y_pred = X_train @ coeffs
    error  = y_pred - y_train
    loss   = np.mean(error ** 2)
    grad   = (2 / n) * (X_train.T @ error)
    coeffs -= lr * grad
    losses.append(loss)

    if (epoch + 1) % 10_000 == 0:
        print(f"Epoch {epoch+1:>6}  |  MSE: {loss:.6f}")

# ── 6. Recover physical coefficients ─────────────────────────────────────────
c1, c2, c3 = coeffs           # [B/J,  -K/J,  -A/J]

J_known = TRUE["J"]           # J must be known externally (e.g. measured inertia)
J_est   =  J_known
B_est   =  c1 * J_est
K_est   = -c2 * J_est
A_est   = -c3 * J_est

# ── 7. Results ────────────────────────────────────────────────────────────────
y_test_pred = X_test @ coeffs
ss_res = np.sum((y_test - y_test_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2     = 1 - ss_res / ss_tot

print("\n" + "=" * 40)
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

# ── 8. Simulate ODE with estimated coefficients ───────────────────────────────
def ode_est(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = (B_est * I_val - K_est * dth - A_est * th) / J_est
    return [dth, d2th]

sol_est   = solve_ivp(ode_est, [t[0], t[-1]], [0.0, 0.0], t_eval=t, max_step=dt)
theta_est = sol_est.y[0]

# ── 9. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
fig.suptitle("ODE system identification via gradient descent", fontsize=13)

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

axes[2].plot(losses, color="darkorange", linewidth=1.5)
axes[2].set_ylabel("MSE loss")
axes[2].set_xlabel("Epoch")
axes[2].set_title("Training loss over epochs")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ode_plot.png", dpi=150)
plt.show()
print("Plot saved to ode_plot.png")