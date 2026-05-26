import archimedes as arc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from archimedes.sysid import Timeseries
from archimedes.observers import ExtendedKalmanFilter
from scipy.signal import butter, filtfilt
from dataclasses import dataclass

SIMLOG = "simlog-20260218_130000"
SUB    = 1

BASE = rf"C:\Users\User\Documents\fbwphlab.py\CSV data_processed\{SIMLOG}\data\aircraft\data"
TICK = rf"C:\Users\User\Documents\fbwphlab.py\CSV data_processed\{SIMLOG}\data\aircraft"

I_x         = pd.read_csv(rf"{BASE}\IservoAil_subfile_{SUB}.csv")
I_y         = pd.read_csv(rf"{BASE}\IservoElev_subfile_{SUB}.csv")
delta_ail   = pd.read_csv(rf"{BASE}\DeltaAil_subfile_{SUB}.csv")
delta_elev  = pd.read_csv(rf"{BASE}\DeltaElev_subfile_{SUB}.csv")
delta_drum_ail  = pd.read_csv(rf"{BASE}\DeltaDrumAil_subfile_{SUB}.csv")
delta_drum_elev = pd.read_csv(rf"{BASE}\DeltaDrumElev_subfile_{SUB}.csv")
true_airspeed   = pd.read_csv(rf"{BASE}\VTrue_subfile_{SUB}.csv")
ticks           = pd.read_csv(rf"{TICK}\tick_subfile_{SUB}.csv")

fs = 1000 #sampling frequency
rho = 1.225

# Convert to numpy
ticks = ticks.to_numpy().flatten()
I_x = I_y.to_numpy().flatten()
I_y = I_y.to_numpy().flatten()
delta_ail = delta_elev.to_numpy().flatten() 
delta_elev = delta_elev.to_numpy().flatten() 
delta_drum_ail = delta_drum_elev.to_numpy().flatten() 
delta_drum_elev = delta_drum_elev.to_numpy().flatten() 
VTrue = true_airspeed.to_numpy().flatten()

#Create time vector
t = (ticks-ticks[0])/10000

# Save raw arrays before any slicing
I_x_raw            = I_x.copy()
delta_drum_ail_raw = delta_drum_ail.copy()
delta_ail_raw      = delta_ail.copy()
VTrue_raw          = VTrue.copy()
t_raw              = t.copy()

#Apply butter filter to smooth out current and drum deflection
b_but, a_but = butter(4, 15/(fs/2))

I_x_filt = filtfilt(b_but, a_but, I_x)
delta_drum_ail_filt = filtfilt(b_but, a_but, delta_drum_ail)
delta_ail_filt = filtfilt(b_but, a_but, delta_ail)
v_true_filt = filtfilt(b_but, a_but, VTrue)

fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

axes[0].plot(t, I_x_filt)
axes[0].set_xlabel('Time [s]')
axes[0].set_ylabel('Current')

axes[1].plot(t, delta_drum_ail)
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Drum Deflection [rad]')
axes[1].legend()

axes[2].plot(t, delta_ail_filt)
axes[2].set_ylabel('Aileron Deflection [rad]')
axes[2].set_xlabel('Time [s]')

axes[3].plot(t, v_true_filt)
axes[3].set_ylabel('Speed [m/s]')
axes[3].set_xlabel('Time [s]')

plt.tight_layout()
plt.show()

idx = (t >= 5) & (t <= 35)

I_x = I_x_filt[idx]
delta_drum_ail = delta_drum_ail_filt[idx]
delta_ail = delta_ail_filt[idx]
VTrue = v_true_filt[idx]
t = t[idx]

dt = t[1] - t[0]

#Current to Drum deflection
t_active = t
u_active = I_x
y_active = delta_drum_ail

#Create Timeseries variable

data = Timeseries(
    ts=t_active,
    us=u_active.reshape(1, -1),
    ys=y_active.reshape(1,-1)
)

@dataclass
class DrumParams:
    J: float #Moment of inertia 
    c: float #Damping coefficient
    k: float #Spring coefficient
    Ki: float #Current 
    ka: float #Aerodynamic stiffness
    ba: float #Aerodynamic damping 

def drum_dynamics(t, x, u, p):

    theta = x[0]
    omega = x[1]

    i = u[0]

    dtheta = omega
    domega = ((p["Ki"]/p["J"]) * i - (p["c"]/p["J"]) * omega - (p["k"]/p["J"]) * theta)

    return np.stack([dtheta, domega])

def obs(t, x, u, p):
    return x[0]

dyn_d = arc.discretize(drum_dynamics, dt)

Q = np.eye(2) * 1e-6
R = np.eye(1) * 1e-4

ekf = arc.observers.ExtendedKalmanFilter(dyn_d, obs, Q, R)

p_guess = {
    "J": 0.01,
    "c": 0.1,
    "k": 1.0,
    "Ki": 1.0
}

x0 = np.array([y_active[0], 0.0])

# Order must match p_guess keys: J, B, K, Kt
lb = {"J": 1e-5, "c": 0.01, "k": 1e-4, "Ki": 1e-4}
ub = {"J": 10, "c": 10, "k": 10, "Ki": 10}

print("Starting PEM...")
result = arc.sysid.pem(ekf, data, p_guess, x0=x0, bounds=(lb, ub))
print("PEM done:", result.p)

estimated_params = result.p

def predict_drum(t_vec, u_vec, x0_drum):
    x, preds = x0_drum.copy(), []
    for i in range(len(t_vec)):
        u = np.array([u_vec[i]])
        preds.append(float(obs(t_vec[i], x, u, estimated_params)))
        x = np.array(dyn_d(t_vec[i], x, u, estimated_params))
    return np.array(preds)

y_pred = predict_drum(t_active, u_active, x0)

correlation_check = np.corrcoef(-y_pred, delta_ail)[0,1]
print(f"Corrected correlation: {correlation_check:.3f}")

# Rigid cable assumption: delta_ail = G * theta_drum + b_bias
A = np.column_stack([-y_pred, np.ones(len(y_pred))])
result_ls = np.linalg.lstsq(A, delta_ail, rcond=None)
G, b_bias = result_ls[0]
print(f"G = {G:.4f}, bias = {b_bias:.4f}")

y_surface_pred = G * (-y_pred) + b_bias

plt.figure(figsize=(12, 4))
plt.plot(t_active, delta_ail, label='Measured Aileron')
plt.plot(t_active, y_surface_pred, label='Predicted Aileron (I → drum → G)', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('Deflection [rad]')
plt.title('End-to-End: Current → Aileron Deflection (Rigid Cable)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

idx_val = (t_raw >=  0) & (t_raw <= 160)

t_val             = t_raw[idx_val]
u_val             = filtfilt(b_but, a_but, I_x_raw[idx_val])
y_val             = filtfilt(b_but, a_but, delta_ail_raw[idx_val])
drum_val_filt     = filtfilt(b_but, a_but, delta_drum_ail_raw[idx_val])

x0_val        = np.array([drum_val_filt[0], 0.0])
y_pred_val    = predict_drum(t_val, u_val, x0_val)
y_surface_val = G * (-y_pred_val) + b_bias

plt.figure(figsize=(12, 4))
plt.plot(t_val, y_val, label='Measured Aileron')
plt.plot(t_val, y_surface_val, label='Predicted (validation)', linestyle='--')
plt.xlabel('Time [s]'); plt.ylabel('Deflection [rad]')
plt.title('Validation: Current → Aileron Deflection')
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

residuals = y_val - y_surface_val
rmse = np.sqrt(np.mean(residuals**2))
r2   = 1 - np.var(residuals) / np.var(y_val)
print(f"Validation RMSE = {rmse:.5f} rad")
print(f"Validation R²   = {r2:.4f}")


def fit_percent(y, yhat):
    denom = np.linalg.norm(y - np.mean(y))
    if denom == 0:
        return np.nan
    return 100 * (1 - np.linalg.norm(y - yhat) / denom)

fit = fit_percent(y_val, y_surface_val)
print(f"the fit is: {fit} %")