import numpy as np
import archimedes as arc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

from get_data import data_grabber



# ============================================
# 1. LOAD DATA
# ============================================

i = 2
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

aileron_df['value'] -= aileron_df['value'].tail(100).mean()
input_df['value'] -= input_df['value'].tail(100).mean()
drum_df['value'] -= drum_df['value'].tail(100).mean()


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
alpha = alpha_df['value'].values.copy()


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

# ===========================================
# CENTERING THE DATA AROUND ZERO
# ===========================================

n_trim = 200   # first 200 samples of the fit window, adjust if needed

# raw windowed signals


# trim values from quiet pre-maneuver part
drum_trim = np.mean(delta_drum[:n_trim])
elev_trim = np.mean(delta_aileron[:n_trim])
input_trim = np.mean(input_command[:n_trim])
alpha_trim = np.mean(alpha[:n_trim])

# centered signals
delta_drum -= drum_trim
delta_aileron -= elev_trim
input_command -= input_trim
alpha -= alpha_trim

print("u0_fit =", input_command[0], q[0], alpha[0])

# ============================================
# 3. DEFINE DYNAMICS (4 states)
# ============================================
fixed = True
delay = False
detrended = True
test = True
window = True
verify = True
start_idx_eval = 0
end_idx_eval = 104000


fixed_params = {
    'J_aileron': 9.70e-02,
    'J_drum': 1.88e-03,
    'aero_damping': 4.50e-04,
    'aero_stiffness': 5.20e-03,
    'drum_friction': 3.70e-01,
    'link_damping': 4.40e+00,
    'link_stiffness': 3.30e+01,
    'torque_gain': 6.75e-01,
}


def actuator_with_linkage(t, x, u, params):
    theta_d, theta_d_dot, theta_a, theta_a_dot = x
    current = u[0]
    q_val = u[1]
    alpha_val = u[2]


    rel0 = 0
    c_alpha = 0
    gamma = 1

    if fixed:
        J_a = 9.59e-02              # J_aileron
        B_a   = 4.44e-04 * q_val    # aero_damping
        K_a = 5.17e-03 * q_val      # aero_stiffness
        B_s   = 4.38e+00            # link_damping
        K_s = 3.30e+01              # link_stiffness
        torque_gain = 6.75e-01      # torque_gain
        J_d = 1.88e-03              # J_drum
        B_d = 3.70e-01              # drum_friction
    else:
        K_s = params['link_stiffness']
        torque_gain = params['torque_gain']
        K_a = params['aero_stiffness'] * q_val
        B_a = params['aero_damping'] * q_val
        J_a = params['J_aileron']
        J_d = params['J_drum']
        B_s = params['link_damping']
        B_d = params['drum_friction']


    I_dead = 0.08
    current_eff = np.sign(current) * np.maximum(np.abs(current) - I_dead, 0.0)
    tau_m = torque_gain * current_eff

    rel = theta_d - theta_a * gamma - rel0
    rel_dot = theta_d_dot - theta_a_dot

    theta_d_ddot = (tau_m - K_s * rel - B_s * rel_dot - B_d * theta_d_dot) / J_d
    theta_a_ddot = (K_s * rel + B_s * rel_dot - K_a * (theta_a + alpha_val*c_alpha) - B_a * theta_a_dot) / J_a

    return np.hstack([theta_d_dot, theta_d_ddot, theta_a_dot, theta_a_ddot])


# ============================================
# 4. MEASUREMENT FUNCTION (only positions)
# ============================================

def measure_positions(t, x, u, params):
    return np.hstack([
        x[0],
        x[2]
    ])

# --- Test the dynamics and measurement ---
sample_t = time_seconds[0]
sample_x = np.array([0.0, 0.0, 0.0, 0.0])
sample_u = np.array([input_command[0], q[0], alpha[0]])
sample_params = {
    'J_drum': 0.01, 'J_aileron': 0.1, 'link_stiffness': 100.0,
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
        }

else:
    params_guess = {
            'J_aileron': 5.3369e-02,
            'J_drum': 3.1532e-02,
            'aero_damping': 4.6338e-04,
            'aero_stiffness': 5.2916e-03,
            'drum_friction': 1.0003e-02,
            'link_damping': 3.2028e+00,
            'link_stiffness': 4.4195e+01,
            'torque_gain': 6.2709e-01
            }



# ============================================
# 8. RUN THE IDENTIFICATION
# ============================================
if window:
    activity = np.abs(np.gradient(input_command))
    event_idx = np.argmax(activity)
    start_idx = max(0, event_idx - 300)
    N_sub = 2000

    time_sub = time[start_idx:start_idx + N_sub]
    q_sub = q[start_idx:start_idx + N_sub]
    ys_sub = ys[:, start_idx:start_idx + N_sub].copy()
    input_sub = input_command[start_idx:start_idx + N_sub]
    alpha_sub = alpha[start_idx:start_idx + N_sub]


else:
    start_time = 1060
    end_time = 1070

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


n_trim = 200  # or 100-300 samples before the event
theta_d0 = np.mean(ys_sub[0, :n_trim])
theta_a0 = np.mean(ys_sub[1, :n_trim])
u0_trim  = np.mean(input_sub[:n_trim])
a0_trim  = np.mean(alpha_sub[:n_trim])

# keep outputs absolute
# ys_sub stays unchanged

# use perturbation inputs around local trim
input_sub = input_sub - u0_trim
alpha_sub = alpha_sub - a0_trim


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
    x0_guess = np.array([theta_d0, 0.0, theta_a0, 0.0], dtype=float)
                          



options = {'max_nfev': 500, 'ftol': 1e-6, 'xtol': 1e-6}





if fixed:
    lower_bounds = {
        }

    upper_bounds = {
        }

    

else:
    lower_bounds = {
        'J_aileron': 9e-4,
        'J_drum': 2e-3,
        'aero_damping': 1e-4,
        'aero_stiffness': 1e-3,
        'drum_friction': 1e-3,
        'link_damping': 5e-1,
        'link_stiffness': 1.5e1,
        'torque_gain': 2e-2,
    }

    upper_bounds = {
        'J_aileron': 2e-1,
        'J_drum': 2e-1,
        'aero_damping': 1e-3,
        'aero_stiffness': 1e-2,
        'drum_friction': 5e-1,
        'link_damping': 5.0,
        'link_stiffness': 8e1,
        'torque_gain': 1.5,
    }


bounds = (lower_bounds, upper_bounds)



def extract_x0_from_result(result, fallback_x0):
    # Try the most likely attribute names safely
    for attr in ['x0', 'x']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            try:
                arr = np.array(val, dtype=float).reshape(-1)
                if arr.size == 4:
                    return arr
            except Exception:
                pass
    return np.array(fallback_x0, dtype=float).copy()


if verify:
    # No free parameters: only estimate x0
    params_guess = {}
    params_test = fixed_params.copy()

    result = arc.sysid.pem(
        ekf,
        data_sub,
        params_guess,
        x0=x0_guess,
        estimate_x0=True,
        options=options,
    )

    x0_est = extract_x0_from_result(result, x0_guess)
    print("Estimated x0 from verify mode:", x0_est)

else:
    # Normal parameter estimation
    result = arc.sysid.pem(
        ekf,
        data_sub,
        params_guess,
        x0=x0_guess,
        estimate_x0=False,
        options=options,
        bounds=bounds
    )

    params_test = dict(result.p)
    x0_est = np.array(x0_guess, dtype=float).copy()




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

    ax2.plot(time_sub, ys_sub[1], 'b-', label='Measured elevator (fit window)')
    ax2.plot(time_sub, outputs_sim_sub[1], 'r--', label='Simulated elevator')
    ax2.set_ylabel('Elevator position (rad)')
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

    ax2.plot(time_seconds, delta_aileron, 'b-', label='Measured elevator')
    ax2.plot(time_seconds, outputs_sim[1], 'r--', label='Simulated elevator')
    ax2.set_ylabel('Elevator position (rad)')
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




def evaluate_params_on_full_dataset(params, start_idx, end_idx, detrended, x0_eval=None, make_plots=True):
    # Full measured outputs
    ys_full = np.vstack([delta_drum, delta_aileron])



    # Initial condition for full-data validation
    if x0_eval is not None:
        x0_full = np.array(x0_eval, dtype=float).copy()
    else:
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
    print(f"Raw full fit - elevator: {fit_ail_raw:.1f}%")
    print(f"Detrended full fit - drum:    {fit_drum_detr:.1f}%")
    print(f"Detrended full fit - elevator: {fit_ail_detr:.1f}%")

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(time_eval, ys_full[0], 'b-', label='Measured drum')
        ax1.plot(time_eval, outputs_full[0], 'r--', label='Simulated drum')
        ax1.set_ylabel('Drum position (rad)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_eval, ys_full[1], 'b-', label='Measured elevator')
        ax2.plot(time_eval, outputs_full[1], 'r--', label='Simulated elevator')
        ax2.set_ylabel('Elevator position (rad)')
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

        ax4.plot(time_eval, ys_full_detr[1], 'b-', label='Measured elevator (detrended)')
        ax4.plot(time_eval, outputs_full_detr[1], 'r--', label='Simulated elevator (detrended)')
        ax4.set_ylabel('Elevator position (rad)')
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
    params_test,
    start_idx_eval,
    end_idx_eval,
    detrended,
    x0_eval=x0_est,
    make_plots=True
)
