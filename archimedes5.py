from matplotlib.pylab import lstsq
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── 1. Generate synthetic data ────────────────────────────────────────────────
# ODE:  J·θ̈  +  K·θ̇  +  A·θ  =  B·I(t)

theta = pd.read_csv("DeltaAil_subfile_4.csv",  skiprows=1).squeeze().to_numpy()
I     = pd.read_csv("IservoAil_subfile_4.csv", skiprows=1).squeeze().to_numpy()
dt    = 0.001            # <-- set your sample period in seconds (e.g. 0.01 for 100Hz)

t = np.arange(len(theta)) * dt

# ── 2. Numerical derivatives ──────────────────────────────────────────────────
theta   = savgol_filter(theta, window_length=100, polyorder=2)
dtheta  = savgol_filter(theta, window_length=100, polyorder=2, deriv=1, delta=dt)
d2theta = savgol_filter(theta, window_length=100, polyorder=2, deriv=2, delta=dt)


fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
fig.suptitle("ODE system identification via gradient descent", fontsize=13)
 
axes[0].plot(t, theta,     label="Measured θ(t)",  color="steelblue", linewidth=2)
axes[0].set_ylabel("θ  (rad)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
 
axes[1].plot(t, dtheta, label="I(t) forcing", color="seagreen", linewidth=1.5)
axes[1].set_ylabel("Current  I(t)")
axes[1].set_xlabel("Time  (s)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
 
axes[2].plot(d2theta, color="darkorange", linewidth=1.5)
axes[2].set_ylabel("MSE loss")
axes[2].set_xlabel("Epoch")
axes[2].set_title("Training loss over epochs")
axes[2].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig("ode_plot.png", dpi=150)
plt.show()
# ── 3. Build feature matrix ───────────────────────────────────────────────────
# Rearranging:  θ̈ = (B/J)·I  -  (K/J)·θ̇  -  (A/J)·θ
# So regress:   θ̈  ~  [I,  θ̇,  θ]   →  coefficients [B/J,  -K/J,  -A/J]

X = np.column_stack([I, dtheta, theta, np.ones(theta.shape)])
y = d2theta

x, residuals, rank, s = lstsq(X, y)

print(x)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False    # shuffle=False keeps time order
)

# ── 5. Gradient descent ───────────────────────────────────────────────────────
# lr is small because features are unscaled — this lets us see the loss
# curve actually descend across many epochs rather than converging in 1 step.
lr     = 0.01
epochs = 5_000
coeffs = np.zeros(4)
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
BoverJ =  x[0]
KoverJ = -x[1]
AoverJ = -x[2]

print(np.roots(x[:3]))

# ── 7. Results ────────────────────────────────────────────────────────────────
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
 
# ── 8. Simulate ODE with estimated ratios ─────────────────────────────────────
def ode_est(t_val, y):
    th, dth = y
    I_val   = np.interp(t_val, t, I)
    d2th    = BoverJ * I_val - KoverJ * dth - AoverJ * th
    return [dth, d2th]
 
sol_est   = solve_ivp(ode_est, [t[0], t[-1]], [theta[0], dtheta[0]], t_eval=t, max_step=dt)
theta_est = sol_est.y[0]
 
# ── 9. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
fig.suptitle("ODE system identification via gradient descent", fontsize=13)
 
axes[0].plot(t, theta,     label="Measured θ(t)",  color="steelblue", linewidth=2)
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