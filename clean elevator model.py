import numpy as np
import archimedes as arc
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from get_data import data_grabber


# ============================================================
# 0. USER SETTINGS / CONFIGURATION
# ============================================================

# -----------------------------
# Dataset selection
# -----------------------------
i = 2
flight_name = "simlog-20260218_130000"
data_type = "aircraft"

# -----------------------------
# Model / identification modes
# -----------------------------
fixed = True          # True  -> use fixed generalized parameter set
                      # False -> estimate parameters from data

verify = True        # True  -> estimate only x0 while keeping parameters fixed
                      # False -> estimate parameters normally

window = True         # True  -> automatically choose fit window from input activity
                      # False -> use manual start/end time window

delay = False         # True  -> add delayed input for fitting
detrended = True      # Used in full-dataset evaluation convention

# -----------------------------
# Simulation / fitting settings
# -----------------------------
dt = 1e-3
delay_samples = 10
start_idx_eval = 0
end_idx_eval = 104000

# Automatic fit window settings
window_pre_samples = 300
window_length = 2000

# Manual fit window settings
manual_start_time = 1060
manual_end_time = 1070

# Trimming / centering settings
tail_trim_samples = 100
local_trim_samples = 200

# Deadzone in current input
I_dead = 0.08

# Preprocess settings for tail-mean centering of signals
center_input_tail = True
center_elevator_tail = True
center_drum_tail = True 
remove_trim = True

# ============================================================
# 1. FIXED PARAMETER SET / INITIAL GUESSES / BOUNDS
# ============================================================

fixed_params = {
    "J_aileron": 9.70e-02,
    "J_drum": 1.88e-03,
    "aero_damping": 4.50e-04,
    "aero_stiffness": 5.20e-03,
    "drum_friction": 3.70e-01,
    "link_damping": 4.40e00,
    "link_stiffness": 3.30e01,
    "torque_gain": 6.75e-01,
}

# If fixed=True, fill this with any parameters you want to estimate (e.g. biases) while keeping the rest fixed.
remaining_params_guess = {}

fixed_bounds_lower = {}

fixed_bounds_upper = {}

fixed_bounds = (fixed_bounds_lower, fixed_bounds_upper)


# Initial guess when estimating parameters (fixed=False).
free_params_guess = {
    "J_aileron": 5.3369e-02,
    "J_drum": 3.1532e-02,
    "aero_damping": 4.6338e-04,
    "aero_stiffness": 5.2916e-03,
    "drum_friction": 1.0003e-02,
    "link_damping": 3.2028e00,
    "link_stiffness": 4.4195e01,
    "torque_gain": 6.2709e-01,
}

lower_bounds = {
    "J_aileron": 9e-4,
    "J_drum": 2e-3,
    "aero_damping": 1e-4,
    "aero_stiffness": 1e-3,
    "drum_friction": 1e-3,
    "link_damping": 5e-1,
    "link_stiffness": 1.5e1,
    "torque_gain": 2e-2,
}

upper_bounds = {
    "J_aileron": 2e-1,
    "J_drum": 2e-1,
    "aero_damping": 1e-3,
    "aero_stiffness": 1e-2,
    "drum_friction": 5e-1,
    "link_damping": 5.0,
    "link_stiffness": 8e1,
    "torque_gain": 1.5,
}

bounds = (lower_bounds, upper_bounds)


# ============================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================

def load_elevator_dataset(flight_name, data_type, subfile_idx):
    
    # Load all required elevator-related signals for one subfile.
    
    input_df, time_df = data_grabber(
        flight_name, data_type, f"IservoElev_subfile_{subfile_idx}", gettick=True
    )
    drum_df, _ = data_grabber(
        flight_name, data_type, f"DeltaDrumElev_subfile_{subfile_idx}", gettick=False
    )
    elev_df, _ = data_grabber(
        flight_name, data_type, f"DeltaElev_subfile_{subfile_idx}", gettick=False
    )
    airspeed_df, _ = data_grabber(
        flight_name, data_type, f"VTrue_subfile_{subfile_idx}", gettick=False
    )
    pressure_df, _ = data_grabber(
        flight_name, data_type, f"StaticPres_subfile_{subfile_idx}", gettick=False
    )
    temperature_df, _ = data_grabber(
        flight_name, data_type, f"StaticTemp_subfile_{subfile_idx}", gettick=False
    )
    alpha_df, _ = data_grabber(
        flight_name, data_type, f"Alpha_subfile_{subfile_idx}", gettick=False
    )

    if time_df is None or input_df is None:
        raise FileNotFoundError("Data grabber failed to find the files.")

    return {
        "input_df": input_df,
        "time_df": time_df,
        "drum_df": drum_df,
        "elev_df": elev_df,
        "airspeed_df": airspeed_df,
        "pressure_df": pressure_df,
        "temperature_df": temperature_df,
        "alpha_df": alpha_df,
    }


def preprocess_signals(data):
    
    # onvert DataFrames to NumPy arrays, apply tail-mean centering and smoothing,
    # and compute dynamic pressure.
    
    input_df = data["input_df"].copy()
    drum_df = data["drum_df"].copy()
    elev_df = data["elev_df"].copy()

    # Tail-mean centering
    if center_elevator_tail:
        elev_df["value"] -= elev_df["value"].tail(tail_trim_samples).mean()
    if center_drum_tail:
        drum_df["value"] -= drum_df["value"].tail(tail_trim_samples).mean()
    if center_input_tail:
        input_df["value"] -= input_df["value"].tail(tail_trim_samples).mean()

    # Time
    time_ticks = data["time_df"]["value"].values
    time_seconds = time_ticks / 10000.0

    # Raw arrays
    input_command = input_df["value"].values
    delta_drum = drum_df["value"].values
    delta_elevator = elev_df["value"].values
    airspeed = data["airspeed_df"]["value"].values
    pressure = data["pressure_df"]["value"].values
    temperature = data["temperature_df"]["value"].values.copy() + 273.15
    alpha = data["alpha_df"]["value"].values.copy()

    # Smooth key signals
    input_command = savgol_filter(input_command, window_length=101, polyorder=2)
    delta_drum = savgol_filter(delta_drum, window_length=101, polyorder=2)
    delta_elevator = savgol_filter(delta_elevator, window_length=101, polyorder=2)

    # Dynamic pressure
    R_air = 287.05
    rho = pressure / (R_air * temperature)
    q = 0.5 * rho * airspeed**2

    if remove_trim:
        # Global centering around zero using first samples of current arrays
        drum_trim = np.mean(delta_drum[:local_trim_samples])
        elev_trim = np.mean(delta_elevator[:local_trim_samples])
        input_trim = np.mean(input_command[:local_trim_samples])
        alpha_trim = np.mean(alpha[:local_trim_samples])

        delta_drum = delta_drum - drum_trim
        delta_elevator = delta_elevator - elev_trim
        input_command = input_command - input_trim
        alpha = alpha - alpha_trim

    return {
        "time_seconds": time_seconds,
        "input_command": input_command,
        "delta_drum": delta_drum,
        "delta_elevator": delta_elevator,
        "airspeed": airspeed,
        "pressure": pressure,
        "temperature": temperature,
        "alpha": alpha,
        "q": q,
    }


# ============================================================
# 3. MODEL DEFINITION
# ============================================================

def actuator_with_linkage(t, x, u, params):
    """
    4-state elevator actuation model:
        x = [theta_d, theta_d_dot, theta_a, theta_a_dot]

    theta_d : drum angle
    theta_a : elevator angle

    Input vector:
        u[0] = servo current
        u[1] = dynamic pressure q
        u[2] = angle of attack alpha
    """
    theta_d, theta_d_dot, theta_a, theta_a_dot = x
    current = u[0]
    q_val = u[1]
    alpha_val = u[2]

    # Optional geometry / trim modifiers
    rel0 = 0.0
    gamma = 1.0
    c_alpha = 0.0

    # Use either fixed generalized parameters or estimated/free parameters
    if fixed:
        J_a = fixed_params["J_aileron"]
        J_d = fixed_params["J_drum"]
        B_a = fixed_params["aero_damping"] * q_val
        K_a = fixed_params["aero_stiffness"] * q_val
        B_s = fixed_params["link_damping"]
        K_s = fixed_params["link_stiffness"]
        B_d = fixed_params["drum_friction"]
        torque_gain = fixed_params["torque_gain"]
    else:
        J_a = params["J_aileron"]
        J_d = params["J_drum"]
        B_a = params["aero_damping"] * q_val
        K_a = params["aero_stiffness"] * q_val
        B_s = params["link_damping"]
        K_s = params["link_stiffness"]
        B_d = params["drum_friction"]
        torque_gain = params["torque_gain"]

    # Deadzone on current
    current_eff = np.sign(current) * np.maximum(np.abs(current) - I_dead, 0.0)
    tau_m = torque_gain * current_eff

    # Relative deformation in the linkage
    rel = theta_d - gamma * theta_a - rel0
    rel_dot = theta_d_dot - theta_a_dot * gamma

    # Drum dynamics
    theta_d_ddot = (
        tau_m
        - K_s * rel
        - B_s * rel_dot
        - B_d * theta_d_dot
    ) / J_d

    # Elevator dynamics
    theta_a_ddot = (
        K_s * rel
        + B_s * rel_dot
        - K_a * (theta_a + c_alpha * alpha_val)
        - B_a * theta_a_dot
    ) / J_a

    return np.hstack([theta_d_dot, theta_d_ddot, theta_a_dot, theta_a_ddot])


def measure_positions(t, x, u, params):
    """
    Measurement model: only drum and elevator positions are measured.
    """
    return np.hstack([x[0], x[2]])


# ============================================================
# 4. IDENTIFICATION / SIMULATION HELPERS
# ============================================================

def select_fit_window(time_seconds, input_command, q, ys, alpha):
    """
    Select either an automatic fit window (based on input activity)
    or a manually defined time window.
    """
    if window:
        activity = np.abs(np.gradient(input_command))
        event_idx = np.argmax(activity)
        start_idx = max(0, event_idx - window_pre_samples)
        end_idx = start_idx + window_length
    else:
        start_idx = np.searchsorted(time_seconds, manual_start_time)
        end_idx = np.searchsorted(time_seconds, manual_end_time)

    time_sub = time_seconds[start_idx:end_idx]
    q_sub = q[start_idx:end_idx]
    ys_sub = ys[:, start_idx:end_idx].copy()
    input_sub = input_command[start_idx:end_idx].copy()
    alpha_sub = alpha[start_idx:end_idx].copy()

    return start_idx, end_idx, time_sub, q_sub, ys_sub, input_sub, alpha_sub


def prepare_fit_window(ys_sub, input_sub, alpha_sub):
    """
    Prepare the fit window for identification.

    Current convention:
    - outputs are shifted so the first sample is zero
    - input and alpha are centered around their local pre-maneuver trim
    """
    ys_sub[0, :] -= ys_sub[0, 0]
    ys_sub[1, :] -= ys_sub[1, 0]

    u0_trim = np.mean(input_sub[:local_trim_samples])
    a0_trim = np.mean(alpha_sub[:local_trim_samples])

    input_sub = input_sub - u0_trim
    alpha_sub = alpha_sub - a0_trim

    return ys_sub, input_sub, alpha_sub


def apply_input_delay(input_sub):
    """
    Apply a simple sample delay to the input.
    """
    input_sub_delayed = np.empty_like(input_sub)
    input_sub_delayed[:delay_samples] = input_sub[0]
    input_sub_delayed[delay_samples:] = input_sub[:-delay_samples]
    return input_sub_delayed


def extract_x0_from_result(result, fallback_x0):
    """
    Safely extract estimated x0 from the Archimedes result object.
    """
    for attr in ["x0", "x"]:
        if hasattr(result, attr):
            val = getattr(result, attr)
            try:
                arr = np.array(val, dtype=float).reshape(-1)
                if arr.size == 4:
                    return arr
            except Exception:
                pass
    return np.array(fallback_x0, dtype=float).copy()


def simulate_dataset(params, time_vec, u_mat, x0, dyn_discrete):
    """
    Simulate the nonlinear discrete-time model over a dataset.
    """
    n_steps = u_mat.shape[1]
    states = np.zeros((4, n_steps))
    outputs = np.zeros((2, n_steps))

    x = x0.copy()
    states[:, 0] = x
    outputs[:, 0] = measure_positions(time_vec[0], x, u_mat[:, 0], params)

    for k in range(1, n_steps):
        t_cur = time_vec[k - 1]
        x = dyn_discrete(t_cur, x, u_mat[:, k - 1], params)
        states[:, k] = x
        outputs[:, k] = measure_positions(time_vec[k], x, u_mat[:, k], params)

    return states, outputs


def fit_percent(y, yhat):
    """
    Matlab-style fit percentage:
        100 * (1 - ||y - yhat|| / ||y - mean(y)||)
    """
    denom = np.linalg.norm(y - np.mean(y))
    if denom == 0:
        return np.nan
    return 100 * (1 - np.linalg.norm(y - yhat) / denom)


def evaluate_params_on_full_dataset(
    params,
    x0_eval,
    time_seconds,
    input_command,
    q,
    alpha,
    delta_drum,
    delta_elevator,
    start_idx,
    end_idx,
    detrended,
    dyn_discrete,
    make_plots=True,
):
    """
    Simulate the full dataset using the chosen parameters and initial state.
    """
    if end_idx is None:
        end_idx = len(time_seconds)

    time_eval = time_seconds[start_idx:end_idx]
    ys_full = np.vstack([delta_drum, delta_elevator])[:, start_idx:end_idx]

    if x0_eval is not None:
        x0_full = np.array(x0_eval, dtype=float).copy()
    else:
        if detrended:
            x0_full = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            x0_full = np.array([
                delta_drum[start_idx],
                0.0,
                delta_elevator[start_idx],
                0.0,
            ], dtype=float)

    if detrended:
        input_full_ref = input_command - input_command[start_idx]
        alpha_full_ref = alpha - alpha[start_idx]
        us_full_eval = np.stack([input_full_ref, q, alpha_full_ref], axis=0)
    else:
        us_full_eval = np.stack([input_command, q, alpha], axis=0)

    us_full_eval = us_full_eval[:, start_idx:end_idx]

    states_full, outputs_full = simulate_dataset(
        params=params,
        time_vec=time_eval,
        u_mat=us_full_eval,
        x0=x0_full,
        dyn_discrete=dyn_discrete,
    )

    # Raw fits
    fit_drum_raw = fit_percent(ys_full[0], outputs_full[0])
    fit_elev_raw = fit_percent(ys_full[1], outputs_full[1])

    # Detrended fits
    ys_full_detr = ys_full - ys_full[:, [0]]
    outputs_full_detr = outputs_full - outputs_full[:, [0]]

    fit_drum_detr = fit_percent(ys_full_detr[0], outputs_full_detr[0])
    fit_elev_detr = fit_percent(ys_full_detr[1], outputs_full_detr[1])

    print("\n=== Full-dataset validation using test-set parameters ===")
    print(f"Raw full fit - drum:      {fit_drum_raw:.1f}%")
    print(f"Raw full fit - elevator:  {fit_elev_raw:.1f}%")
    print(f"Detrended full fit - drum:     {fit_drum_detr:.1f}%")
    print(f"Detrended full fit - elevator: {fit_elev_detr:.1f}%")

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(time_eval, ys_full[0], "b-", label="Measured drum")
        ax1.plot(time_eval, outputs_full[0], "r--", label="Simulated drum")
        ax1.set_ylabel("Drum position (rad)")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time_eval, ys_full[1], "b-", label="Measured elevator")
        ax2.plot(time_eval, outputs_full[1], "r--", label="Simulated elevator")
        ax2.set_ylabel("Elevator position (rad)")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        ax2.grid(True)
        plt.show()

        fig, (ax3, ax4) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax3.plot(time_eval, ys_full_detr[0], "b-", label="Measured drum (detrended)")
        ax3.plot(time_eval, outputs_full_detr[0], "r--", label="Simulated drum (detrended)")
        ax3.set_ylabel("Drum position (rad)")
        ax3.legend()
        ax3.grid(True)

        ax4.plot(time_eval, ys_full_detr[1], "b-", label="Measured elevator (detrended)")
        ax4.plot(time_eval, outputs_full_detr[1], "r--", label="Simulated elevator (detrended)")
        ax4.set_ylabel("Elevator position (rad)")
        ax4.set_xlabel("Time (s)")
        ax4.legend()
        ax4.grid(True)
        plt.show()

    return {
        "x0_full": x0_full,
        "states_full": states_full,
        "outputs_full": outputs_full,
        "fit_drum_raw": fit_drum_raw,
        "fit_elev_raw": fit_elev_raw,
        "fit_drum_detr": fit_drum_detr,
        "fit_elev_detr": fit_elev_detr,
    }


# ============================================================
# 5. LOAD DATA
# ============================================================

data = load_elevator_dataset(flight_name, data_type, i)
signals = preprocess_signals(data)

time_seconds = signals["time_seconds"]
input_command = signals["input_command"]
delta_drum = signals["delta_drum"]
delta_elevator = signals["delta_elevator"]
airspeed = signals["airspeed"]
pressure = signals["pressure"]
temperature = signals["temperature"]
alpha = signals["alpha"]
q = signals["q"]

print(time_seconds[:5])
print("u0_fit =", input_command[0], q[0], alpha[0])


# ============================================================
# 6. SANITY CHECK: TEST DYNAMICS / MEASUREMENT
# ============================================================

sample_t = time_seconds[0]
sample_x = np.array([0.0, 0.0, 0.0, 0.0])
sample_u = np.array([input_command[0], q[0], alpha[0]])
sample_params = fixed_params.copy() if fixed else free_params_guess.copy()

test_out = actuator_with_linkage(sample_t, sample_x, sample_u, sample_params)
test_meas = measure_positions(sample_t, sample_x, sample_u, sample_params)

print("Dynamics test output:", test_out, "shape:", test_out.shape)
print("Measurement test output:", test_meas, "shape:", test_meas.shape)


# ============================================================
# 7. BUILD DISCRETE MODEL AND EKF
# ============================================================

dyn_discrete = arc.discretize(actuator_with_linkage, dt, method="rk4")
print(f"dt = {dt}")

Q = np.diag([1e-5, 1e-3, 1e-5, 1e-3])
noise_std_drum = 2e-4
noise_std_ail = 5e-4
R = np.diag([noise_std_drum**2, noise_std_ail**2])

ekf = arc.observers.ExtendedKalmanFilter(dyn_discrete, measure_positions, Q, R)


# ============================================================
# 8. BUILD TRAINING DATASET
# ============================================================

us = np.stack([input_command, q, alpha], axis=0)
ys = np.array([delta_drum, delta_elevator])

time = np.ascontiguousarray(time_seconds, dtype=np.float64)
us = np.ascontiguousarray(us, dtype=np.float64)
ys = np.ascontiguousarray(ys, dtype=np.float64)

start_idx, end_idx, time_sub, q_sub, ys_sub, input_sub, alpha_sub = select_fit_window(
    time_seconds=time_seconds,
    input_command=input_command,
    q=q,
    ys=ys,
    alpha=alpha,
)

ys_sub, input_sub, alpha_sub = prepare_fit_window(ys_sub, input_sub, alpha_sub)

if delay:
    input_sub = apply_input_delay(input_sub)

us_sub = np.stack([input_sub, q_sub, alpha_sub], axis=0)
data_sub = arc.sysid.Timeseries(ts=time_sub, us=us_sub, ys=ys_sub)

x0_guess = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
options = {"max_nfev": 500, "ftol": 1e-6, "xtol": 1e-6}


# ============================================================
# 9. RUN IDENTIFICATION / VERIFY x0
# ============================================================

if verify:
    # Estimate only the initial condition while keeping the parameters fixed
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
    sim_params = params_test
    sim_x0 = x0_est

    print("Estimated x0 from verify mode:", x0_est)

else:
    if fixed:
        # No parameter estimation, just use fixed parameters directly
        params_guess = remaining_params_guess.copy()

        result = arc.sysid.pem(
            ekf,
            data_sub,
            params_guess,
            x0=x0_guess,
            estimate_x0=False,
            options=options,
            bounds=fixed_bounds,
        )

        params_test = dict(result.p)
        sim_params = params_test
        sim_x0 = x0_guess.copy()

    else:
        # Estimate parameters normally
        params_guess = free_params_guess.copy()

        result = arc.sysid.pem(
            ekf,
            data_sub,
            params_guess,
            x0=x0_guess,
            estimate_x0=False,
            options=options,
            bounds=bounds,
        )

        params_test = dict(result.p)
        sim_params = params_test
        sim_x0 = x0_guess.copy()


# ============================================================
# 10. DIAGNOSTICS
# ============================================================

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

dx0 = actuator_with_linkage(time_sub[0], sim_x0, us_sub[:, 0], sim_params)
x1_rk4 = dyn_discrete(time_sub[0], sim_x0, us_sub[:, 0], sim_params)
x1_euler = sim_x0 + dt * dx0

print("dx0 =", dx0)
print("x1_rk4 =", x1_rk4)
print("x1_euler =", x1_euler)


# ============================================================
# 11. REPORT PARAMETERS
# ============================================================

print("\n=== Parameters used for simulation ===")
for key, val in sim_params.items():
    print(f"{key}: {val:.4e}")


# ============================================================
# 12. SIMULATE FIT WINDOW
# ============================================================

states_sim_sub, outputs_sim_sub = simulate_dataset(
    params=sim_params,
    time_vec=time_sub,
    u_mat=us_sub,
    x0=sim_x0,
    dyn_discrete=dyn_discrete,
)

print(f"Fit-window simulation complete for {us_sub.shape[1]} time steps")


# ============================================================
# 13. PLOT FIT WINDOW
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

ax1.plot(time_sub, ys_sub[0], "b-", label="Measured drum (fit window)")
ax1.plot(time_sub, outputs_sim_sub[0], "r--", label="Simulated drum")
ax1.set_ylabel("Drum position (rad)")
ax1.legend()
ax1.grid(True)

ax2.plot(time_sub, ys_sub[1], "b-", label="Measured elevator (fit window)")
ax2.plot(time_sub, outputs_sim_sub[1], "r--", label="Simulated elevator")
ax2.set_ylabel("Elevator position (rad)")
ax2.set_xlabel("Time (s)")
ax2.legend()
ax2.grid(True)

plt.show()


# ============================================================
# 14. FIT METRICS FOR FIT WINDOW
# ============================================================

fit_drum = fit_percent(ys_sub[0], outputs_sim_sub[0])
fit_elevator = fit_percent(ys_sub[1], outputs_sim_sub[1])

print(f"\nFit for drum: {fit_drum:.1f}%")
print(f"Fit for elevator: {fit_elevator:.1f}%")


# ============================================================
# 15. FULL-DATASET VALIDATION
# ============================================================

full_evaluation_results = evaluate_params_on_full_dataset(
    params=sim_params,
    x0_eval=sim_x0,
    time_seconds=time_seconds,
    input_command=input_command,
    q=q,
    alpha=alpha,
    delta_drum=delta_drum,
    delta_elevator=delta_elevator,
    start_idx=start_idx_eval,
    end_idx=end_idx_eval,
    detrended=detrended,
    dyn_discrete=dyn_discrete,
    make_plots=True,
)