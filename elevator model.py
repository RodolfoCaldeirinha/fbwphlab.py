import numpy as np
import archimedes as arc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

from get_data import data_grabber



# ============================================
# 1. LOAD DATA
# ============================================

i = 1
flight_name = "simlog-20260218_130000"
data_type = "aircraft"

# Get the first variable AND the time (tick) array simultaneously
input_df, time_df = data_grabber(flight_name, data_type, f"IservoElev_subfile_{i}", gettick=True)

# For the rest, we only need the data (gettick=False), so we ignore the second output with '_'
drum_df, _ = data_grabber(flight_name, data_type, f"DeltaDrumElev_subfile_{i}", gettick=False)
aileron_df, _ = data_grabber(flight_name, data_type, f"DeltaElev_subfile_{i}", gettick=False)
airspeed_df, _ = data_grabber(flight_name, data_type, f"VTrue_subfile_{i}", gettick=False)
pressure_df, _ = data_grabber(flight_name, data_type, f"StaticPres_subfile_{i}", gettick=False)
temperature_df, _ = data_grabber(flight_name, data_type, f"StaticTemp_subfile_{i}", gettick=False)
alpha_df, _ = data_grabber(flight_name, data_type, f"Alpha_subfile_{i}", gettick=False)

if time_df is None or input_df is None:
    raise FileNotFoundError("Data grabber failed to find the files. Your shit is wrong !")

#aileron_df['value'] -= aileron_df['value'].tail(100).mean()
#input_df['value'] -= input_df['value'].tail(100).mean()
#drum_df['value'] -= drum_df['value'].tail(100).mean()


time = time_df['value'].values
time_seconds = time / 10000  # convert from ticks to seconds
input_command = input_df['value'].values
delta_drum = drum_df['value'].values
delta_aileron = aileron_df['value'].values
airspeed = airspeed_df['value'].values
pressure = pressure_df['value'].values
temperature = temperature_df['value'].values
temperature = temperature.copy()
temperature += 273.15
alpha = alpha_df['value'].values


input_command = savgol_filter(input_df['value'].values, window_length=101, polyorder=2)
delta_drum = savgol_filter(drum_df['value'].values, window_length=101, polyorder=2)
delta_aileron = savgol_filter(aileron_df['value'].values, window_length=101, polyorder=2)

#delta_drum = -delta_drum

print(time_seconds[:5])
# ============================================
# 2. COMPUTE DYNAMIC PRESSURE q(t)
# ============================================

R_air = 287.05
rho = pressure / (R_air * temperature)   # density [kg/m³]
q = 0.5 * rho * airspeed**2              # dynamic pressure [Pa]

# ============================================
# 3. DEFINE DYNAMICS (4 states)
# ============================================
fixed = False
delay = False
detrended = True
test = True
window = False
start_idx_eval = 0
end_idx_eval = len(time_seconds)


def actuator_with_linkage(t, x, u, params):
    theta_d, theta_d_dot, theta_a, theta_a_dot = x
    current = u[0]
    q_val = u[1]
    alpha_val = u[2]
    c_alpha = 0

    B_d = params['drum_friction']

    if fixed:
        K_s = 3.1e1           # fixed link_stiffness
        torque_gain = 7.9e-1  # fixed torque_gain
        K_a = 3.3e-3 * q_val  # fixed aero_stiffness
        B_a = 3.4e-4 * q_val  # fixed aero_damping
        J_a = 4.7e-2          # fixed J_aileron
        J_d = 8.9e-2          # fixed J_drum
        B_s = 1.5e0           # fixed link_damping 
    else:
        K_s = params['link_stiffness']
        torque_gain = params['torque_gain']
        K_a = params['aero_stiffness'] * q_val
        B_a = params['aero_damping'] * q_val
        J_a = params['J_aileron']
        J_d = params['J_drum']
        B_s = params['link_damping']


    I_dead = 0.02
    current_eff = np.sign(current) * np.maximum(np.abs(current) - I_dead, 0.0)
    tau_m = torque_gain * current_eff

    rel = theta_d - theta_a
    rel_dot = theta_d_dot - theta_a_dot

    theta_d_ddot = (tau_m - K_s * rel - B_s * rel_dot - B_d * theta_d_dot) / J_d
    theta_a_ddot = (K_s * rel + B_s * rel_dot - K_a * (theta_a + alpha_val*c_alpha) - B_a * theta_a_dot) / J_a

    return np.hstack([theta_d_dot, theta_d_ddot, theta_a_dot, theta_a_ddot])


# ============================================
# 4. MEASUREMENT FUNCTION (only positions)
# ============================================

def measure_positions(t, x, u, params):
    return np.hstack([
        x[0] + params['drum_bias'],
        x[2] + params['aileron_bias']
    ])

# --- Test the dynamics and measurement ---
sample_t = time_seconds[0]
sample_x = np.array([0.0, 0.0, 0.0, 0.0])
sample_u = np.array([input_command[0], q[0], alpha[0]])
sample_params = {
    'drum_bias': 0, 'aileron_bias': 0, 'J_drum': 0.01, 'J_aileron': 0.1, 'link_stiffness': 100.0,
    'link_damping': 1.0, 'drum_friction': 0.1, 'torque_gain': 1.0,
    'aero_stiffness': 0.05, 'aero_damping': 0.02
}
test_out = actuator_with_linkage(sample_t, sample_x, sample_u, sample_params)
print("Dynamics test output:", test_out, "shape:", test_out.shape)

test_meas = measure_positions(sample_t, sample_x, sample_u, sample_params)
print("Measurement test output:", test_meas, "shape:", test_meas.shape)


# ============================================
# 5. SET UP THE IDENTIFICATION
# ============================================
dt = 1e-3
dyn_discrete = arc.discretize(actuator_with_linkage, dt, method="rk4")
print(f'dt = {dt}')
# Noise matrices
Q = np.diag([1e-5, 1e-3, 1e-5, 1e-3])
noise_std_drum = 2e-4
noise_std_ail = 5e-4
R = np.diag([noise_std_drum**2, noise_std_ail**2])

ekf = arc.observers.ExtendedKalmanFilter(dyn_discrete, measure_positions, Q, R)


# ============================================
# 6. PREPARE THE DATA
# ============================================
us = np.stack([input_command, q, alpha], axis=0)        # shape (2, N)
ys = np.array([delta_drum, delta_aileron])       # shape (2, N)

time = np.ascontiguousarray(time_seconds, dtype=np.float64)
us = np.ascontiguousarray(us, dtype=np.float64)
ys = np.ascontiguousarray(ys, dtype=np.float64)

data = arc.sysid.Timeseries(ts=time_seconds, us=us, ys=ys)


# ============================================
# 7. INITIAL PARAMETER GUESSES
# ============================================

if fixed:
    params_guess = {
        'drum_bias': delta_drum[start_idx_eval],
        'aileron_bias': delta_aileron[start_idx_eval],
        'drum_friction': 5e-02,
        }

else:
    params_guess = {
        'drum_bias': delta_drum[start_idx_eval],
        'aileron_bias': delta_aileron[start_idx_eval],
        'J_aileron': 3.1416e-02,
        'J_drum': 1.0455e-01,
        'aero_damping': 2.7328e-04,
        'aero_stiffness': 2.9755e-03,
        'drum_friction': 5e-02,
        'link_damping': 2.1163e+00,
        'link_stiffness': 3.9388e+01,
        'torque_gain': 7.7989e-01,
        }



# ============================================
# 8. RUN THE IDENTIFICATION
# ============================================
if window:
    activity = np.abs(np.gradient(delta_drum)) + np.abs(np.gradient(delta_aileron))
    event_idx = np.argmax(activity)
    start_idx = max(0, event_idx - 1000)
    N_sub = 2000

    time_sub = time[start_idx:start_idx + N_sub]
    q_sub = q[start_idx:start_idx + N_sub]
    ys_sub = ys[:, start_idx:start_idx + N_sub].copy()
    input_sub = input_command[start_idx:start_idx + N_sub]
    alpha_sub = alpha[start_idx:start_idx + N_sub]

else:
    start_time = 991.0
    end_time = 995

    start_idx = np.searchsorted(time_seconds, start_time)
    end_idx = np.searchsorted(time_seconds, end_time)

    time_sub = time[start_idx:end_idx]
    q_sub = q[start_idx:end_idx]
    ys_sub = ys[:, start_idx:end_idx].copy()
    input_sub = input_command[start_idx:end_idx]
    alpha_sub = alpha[start_idx:end_idx]



if detrended:
    ys_sub[0, :] -= ys_sub[0, 0]
    ys_sub[1, :] -= ys_sub[1, 0]
    input_sub = input_sub - input_sub[0]
    alpha_sub = alpha_sub - alpha_sub[0]





if delay:
    delay_samples = 10

    input_sub_delayed = np.empty_like(input_sub)
    input_sub_delayed[:delay_samples] = input_sub[0]
    input_sub_delayed[delay_samples:] = input_sub[:-delay_samples]

    us_sub = np.stack([input_sub_delayed, q_sub, alpha_sub], axis=0)
else:
    us_sub = np.stack([input_sub, q_sub, alpha_sub], axis=0)


data_sub = arc.sysid.Timeseries(ts=time_sub, us=us_sub, ys=ys_sub)

if detrended:
    x0_guess = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
else:
    x0_guess = np.array([
        ys_sub[0, 0],
        0.0,
        ys_sub[1, 0],
        0.0
        ], dtype=float)


options = {'max_nfev': 500, 'ftol': 1e-6, 'xtol': 1e-6}


if fixed:
    lower_bounds = {
        'drum_bias': -1e-1,
        'aileron_bias': -1e-1,
        'drum_friction': 0,
        }

    upper_bounds = {
        'drum_bias': 1e-1,
        'aileron_bias': 1e-1,
        'drum_friction': 1e-0,
        }

    bounds = (lower_bounds, upper_bounds)

else:
    lower_bounds = {
        'drum_bias': -1e-2,
        'aileron_bias': -1e-2,
        'J_aileron': 5e-3,
        'J_drum': 1e-2,
        'aero_damping': 1e-5,
        'aero_stiffness': 1e-4,
        'drum_friction': 0.0,
        'link_damping': 0.1,
        'link_stiffness': 5.0,
        'torque_gain': 1e-2,
        }

    upper_bounds = {
        'drum_bias': 1e-2,
        'aileron_bias': 1e-2,
        'J_aileron': 2e-1,
        'J_drum': 5e-1,
        'aero_damping': 5e-3,
        'aero_stiffness': 2e-2,
        'drum_friction': 1.0,
        'link_damping': 10.0,
        'link_stiffness': 200.0,
        'torque_gain': 5.0,
        }

    bounds = (lower_bounds, upper_bounds)



result = arc.sysid.pem(ekf,
                       data_sub, params_guess, 
                       x0=x0_guess, 
                       estimate_x0=False, 
                       options=options, 
                       bounds=bounds
                       )

params_test = dict(result.p)


print("x0_guess =", x0_guess)
print("u0 =", us[:, 0])
print("input_command[0] =", input_command[0])
print("q[0] =", q[0])

print("All finite?")
print("input_command:", np.isfinite(input_command).all())
print("q:", np.isfinite(q).all())
print("pressure:", np.isfinite(pressure).all())
print("temperature:", np.isfinite(temperature).all())
print("airspeed:", np.isfinite(airspeed).all())

print("Ranges:")
print("input_command min/max:", np.nanmin(input_command), np.nanmax(input_command))
print("pressure min/max:", np.nanmin(pressure), np.nanmax(pressure))
print("temperature min/max:", np.nanmin(temperature), np.nanmax(temperature))
print("airspeed min/max:", np.nanmin(airspeed), np.nanmax(airspeed))
print("q min/max:", np.nanmin(q), np.nanmax(q))


print("u0_fit =", us_sub[:, 0])


dx0 = actuator_with_linkage(time_sub[0], x0_guess, us_sub[:, 0], result.p)
print("dx0 =", dx0)

x1_rk4 = dyn_discrete(time_sub[0], x0_guess, us_sub[:, 0], result.p)
print("x1_rk4 =", x1_rk4)

x1_euler = x0_guess + dt * dx0
print("x1_euler =", x1_euler)


# ============================================
# 9. RESULTS
# ============================================


print("\n=== Estimated Parameters ===")
for key, val in result.p.items():
    print(f"{key}: {val:.4e}")

print("Simulating with estimated parameters...")

# =========================
# A) SIMULATE FIT WINDOW
# =========================

if test:
    n_steps_sub = us_sub.shape[1]
    states_sim_sub = np.zeros((4, n_steps_sub))
    outputs_sim_sub = np.zeros((2, n_steps_sub))

    x_sim_sub = x0_guess.copy()
    states_sim_sub[:, 0] = x_sim_sub
    outputs_sim_sub[:, 0] = measure_positions(time_sub[0], x_sim_sub, us_sub[:, 0], result.p)

    for k in range(1, n_steps_sub):
        t_cur = time_sub[k-1]
        x_sim_sub = dyn_discrete(t_cur, x_sim_sub, us_sub[:, k-1], result.p)
        states_sim_sub[:, k] = x_sim_sub
        outputs_sim_sub[:, k] = measure_positions(time_sub[k], x_sim_sub, us_sub[:, k], result.p)

    print(f"Fit-window simulation complete for {n_steps_sub} time steps")

# =========================
# B) OPTIONAL: SIMULATE FULL DATA
# =========================

if not test:
    n_steps = us.shape[1]
    states_sim = np.zeros((4, n_steps))
    outputs_sim = np.zeros((2, n_steps))

    x0_full = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    x_sim = x0_full.copy()
    states_sim[:, 0] = x_sim
    outputs_sim[:, 0] = measure_positions(time_seconds[0], x_sim, us[:, 0], result.p)

    for k in range(1, n_steps):
        t_cur = time_seconds[k-1]
        x_sim = dyn_discrete(t_cur, x_sim, us[:, k-1], result.p)
        states_sim[:, k] = x_sim
        outputs_sim[:, k] = measure_positions(time_seconds[k], x_sim, us[:, k], result.p)

    print(f"Full simulation complete for {n_steps} time steps")

# ============================================
# 10. PLOT COMPARISON
# ============================================


if test:
    # Plot the FIT WINDOW when testing
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(time_sub, ys_sub[0], 'b-', label='Measured drum (fit window)')
    ax1.plot(time_sub, outputs_sim_sub[0], 'r--', label='Simulated drum')
    ax1.set_ylabel('Drum position (rad)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_sub, ys_sub[1], 'b-', label='Measured aileron (fit window)')
    ax2.plot(time_sub, outputs_sim_sub[1], 'r--', label='Simulated aileron')
    ax2.set_ylabel('Aileron position (rad)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)

    plt.show()

else:
    # Plot the FULL DATA only when not in test mode
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(time_seconds, delta_drum, 'b-', label='Measured drum')
    ax1.plot(time_seconds, outputs_sim[0], 'r--', label='Simulated drum')
    ax1.set_ylabel('Drum position (rad)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_seconds, delta_aileron, 'b-', label='Measured aileron')
    ax2.plot(time_seconds, outputs_sim[1], 'r--', label='Simulated aileron')
    ax2.set_ylabel('Aileron position (rad)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)

    plt.show()


# Fit metrics

if test:
    fit_drum = 100 * (
        1 - np.linalg.norm(ys_sub[0] - outputs_sim_sub[0]) /
        np.linalg.norm(ys_sub[0] - np.mean(ys_sub[0]))
    )
    fit_ail = 100 * (
        1 - np.linalg.norm(ys_sub[1] - outputs_sim_sub[1]) /
        np.linalg.norm(ys_sub[1] - np.mean(ys_sub[1]))
    )
else:
    fit_drum = 100 * (
        1 - np.linalg.norm(delta_drum - outputs_sim[0]) /
        np.linalg.norm(delta_drum - np.mean(delta_drum))
    )
    fit_ail = 100 * (
        1 - np.linalg.norm(delta_aileron - outputs_sim[1]) /
        np.linalg.norm(delta_aileron - np.mean(delta_aileron))
    )



print(f"\nFit for drum: {fit_drum:.1f}%")
print(f"Fit for aileron: {fit_ail:.1f}%")




def simulate_dataset(params, time_vec, u_mat, x0, start_idx, end_idx):
    if end_idx is None:
        end_idx = u_mat.shape[1]

    time_eval = time_vec[start_idx:end_idx]
    u_eval = u_mat[:, start_idx:end_idx]

    n_steps = u_eval.shape[1]
    states = np.zeros((4, n_steps))
    outputs = np.zeros((2, n_steps))

    x = x0.copy()
    states[:, 0] = x
    outputs[:, 0] = measure_positions(time_eval[0], x, u_eval[:, 0], params)

    for k in range(1, n_steps):
        t_cur = time_eval[k - 1]
        x = dyn_discrete(t_cur, x, u_eval[:, k - 1], params)
        states[:, k] = x
        outputs[:, k] = measure_positions(time_eval[k], x, u_eval[:, k], params)

    return time_eval, states, outputs


def fit_percent(y, yhat):
    denom = np.linalg.norm(y - np.mean(y))
    if denom == 0:
        return np.nan
    return 100 * (1 - np.linalg.norm(y - yhat) / denom)




def evaluate_params_on_full_dataset(params, start_idx, end_idx, detrended, make_plots=True):
    # Full measured outputs
    ys_full = np.vstack([delta_drum, delta_aileron])



    # Better than starting from zero for full-data validation
    if detrended:
        x0_full = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        x0_full = np.array([
            delta_drum[start_idx],
            0.0,
            delta_aileron[start_idx],
            0.0
        ], dtype=float)

    # Simulate full dataset with fixed params
    # Use the same input convention as in training

    if detrended:
        input_full_ref = input_command - input_command[start_idx]
        alpha_full_ref = alpha - alpha[start_idx]
        us_full_eval = np.stack([input_full_ref, q, alpha_full_ref], axis=0)
    else:
        us_full_eval = np.stack([input_command, q, alpha], axis=0)



    time_eval, states_full, outputs_full = simulate_dataset(
        params, time_seconds, us_full_eval, x0_full, start_idx, end_idx
        )
    
    ys_full = np.vstack([delta_drum, delta_aileron])[:, start_idx:end_idx]

    # Raw fits
    fit_drum_raw = fit_percent(ys_full[0], outputs_full[0])
    fit_ail_raw = fit_percent(ys_full[1], outputs_full[1])

    # Detrended fits
    ys_full_detr = ys_full - ys_full[:, [0]]
    outputs_full_detr = outputs_full - outputs_full[:, [0]]

    fit_drum_detr = fit_percent(ys_full_detr[0], outputs_full_detr[0])
    fit_ail_detr = fit_percent(ys_full_detr[1], outputs_full_detr[1])

    print("\n=== Full-dataset validation using test-set parameters ===")
    print(f"Raw full fit - drum:    {fit_drum_raw:.1f}%")
    print(f"Raw full fit - aileron: {fit_ail_raw:.1f}%")
    print(f"Detrended full fit - drum:    {fit_drum_detr:.1f}%")
    print(f"Detrended full fit - aileron: {fit_ail_detr:.1f}%")

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(time_eval, ys_full[0], 'b-', label='Measured drum')
        ax1.plot(time_eval, outputs_full[0], 'r--', label='Simulated drum')
        ax1.set_ylabel('Drum position (rad)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_eval, ys_full[1], 'b-', label='Measured aileron')
        ax2.plot(time_eval, outputs_full[1], 'r--', label='Simulated aileron')
        ax2.set_ylabel('Aileron position (rad)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True)

        plt.show()

        fig, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax3.plot(time_eval, ys_full_detr[0], 'b-', label='Measured drum (detrended)')
        ax3.plot(time_eval, outputs_full_detr[0], 'r--', label='Simulated drum (detrended)')
        ax3.set_ylabel('Drum position (rad)')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(time_eval, ys_full_detr[1], 'b-', label='Measured aileron (detrended)')
        ax4.plot(time_eval, outputs_full_detr[1], 'r--', label='Simulated aileron (detrended)')
        ax4.set_ylabel('Aileron position (rad)')
        ax4.set_xlabel('Time (s)')
        ax4.legend()
        ax4.grid(True)

        plt.show()

        '''
        e_drum = ys_full_detr[0] - outputs_full_detr[0]

        plt.figure(figsize=(10,4))
        plt.plot(time_eval, e_drum)
        plt.grid(True)
        plt.ylabel("Drum error (rad)")
        plt.xlabel("Time (s)")
        plt.title("Detrended drum error")
        plt.show()

        print("Mean drum error:", np.mean(e_drum))
        print("RMS drum error:", np.sqrt(np.mean(e_drum**2)))

        e_drum = ys_full_detr[0] - outputs_full_detr[0]
        drum_offset_corr = np.mean(e_drum)

        outputs_full_detr_corr = outputs_full_detr.copy()
        outputs_full_detr_corr[0] += drum_offset_corr

        fit_drum_detr_corr = fit_percent(ys_full_detr[0], outputs_full_detr_corr[0])2


        print("Corrected detrended drum fit:", fit_drum_detr_corr)
        print("Applied drum offset correction:", drum_offset_corr)

        '''

    return {
        'x0_full': x0_full,
        'states_full': states_full,
        'outputs_full': outputs_full,
        'fit_drum_raw': fit_drum_raw,
        'fit_ail_raw': fit_ail_raw,
        'fit_drum_detr': fit_drum_detr,
        'fit_ail_detr': fit_ail_detr,
    }

full_evaluation_results = evaluate_params_on_full_dataset(
    params_test, start_idx_eval, end_idx_eval, detrended, make_plots=True
)

