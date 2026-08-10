import numpy as np                          # standard numerical arrays
import archimedes as arc                    # Archimedes: ODE solver + autodiff optimiser
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
import scipy

def convert_array(file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return [float(line.strip()) for line in lines[1:]]  # skip header, convert all

t_begin = 982.7500 #[s]
t_end = 1910.1990 #[s]


def butter_filter(x, filtering_order = 2, cut_off = 15, filter_type = 'low'):
    def convert_array(file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return [float(line.strip()) for line in lines[1:]]  # skip header, convert all
    
    filtering_order = 2
    cut_off = 15
    filter_type = 'low'
    b, a = scipy.signal.butter(N=filtering_order, Wn=cut_off, btype=filter_type, fs=1000) #(order, cutoff frequency, filter type(low or hgih), sampling frequency)

    x_filtered = scipy.signal.filtfilt(b, a, x)
    return x_filtered


theta_meas = convert_array("Master/Master/data_toplevel/CSV data/simlog-20250701_111742/data/aircraft/data/DeltaDrumAil.csv")
t_data = np.linspace(t_begin, t_end, len(theta_meas))
theta_meas = butter_filter(theta_meas)
theta_dot_meas = np.gradient(theta_meas, t_data)
I_data = convert_array("Master/Master/data_toplevel/CSV data/simlog-20250701_111742/data/aircraft/data/IservoAil.csv")
I_func = interp1d(t_data, I_data,kind="linear",fill_value="extrapolate",)

#The DE: A*theta_ddot + B*theta_dot + C*theta = D*I(t)

def dynamics(t, s, params):
    A = params[0]                           # inertia coefficient
    B = params[1]                           # damping coefficient
    C = params[2]                           # stiffness coefficient
    D = params[3]                           # motor torque constant
 
    theta = s[0]                            # current aileron angle from state
    theta_dot = s[1]                            # current angular rate from state
    I_t   = I_func(t)               # interpolate current I at this time step
 
    theta_ddot = (D * I_t - B * theta_dot - C * theta) / A  # solve for angular acceleration
 
    return np.hstack([theta_dot, theta_ddot])

def simulate(params):
    """
    Integrate the ODE from t=0 to t=T using the given parameters.
    Returns the simulated state trajectory, shape (2, N):
      row 0 → predicted x(t)
      row 1 → predicted x_dot(t)
    """
    theta_0 = np.array([theta_meas[0], theta_dot_meas[0]])  # initial state taken from first data sample
 
    ode_fn = lambda t, s: dynamics(t, s, params)  # wrap dynamics so it only takes (t, s)

    xs = arc.odeint(                        # Archimedes ODE integrator (RK45 by default)
        ode_fn,                             # the dynamics function f(t, s)
        t_span=(t_data[0], t_data[-1]),     # integrate from first to last timestamp
        x0=theta_0,                              # initial conditions [x0, x_dot0]
        t_eval=t_data,                      # return solution at every measurement timestamp
    )
    return xs                               # shape: (2, len(t_data))

def residuals(params):
    """
    Compute the vector of residuals between simulation and flight data.
    Archimedes differentiates through this function automatically
    to compute the Jacobian needed by the Levenberg-Marquardt optimiser.
    """
    xs_sim   = simulate(params)             # run the ODE forward with current parameters
 
    x_sim    = xs_sim[0, :]                 # simulated aileron angle at each timestep
    xdot_sim = xs_sim[1, :]                 # simulated angular rate at each timestep
 
    res_x    = x_sim    - theta_meas           # angle residuals (predicted - measured)
    res_xdot = xdot_sim - theta_dot_meas        # rate residuals  (predicted - measured)
 
    # Stack both residual vectors into one flat array for the optimiser
    # If you only have x measured, replace with: return res_x
    return np.hstack([res_x, res_xdot])    # combined residual vector, length 2*N

params0 = np.array([1.0, 0.5, 2.0, 1.0])  # initial guess for [A, B, C, D]
 
result = arc.optimize.lm_solve(            # Levenberg-Marquardt nonlinear least-squares
    residuals,                             # the residual function to minimise
    params0,                               # starting point for optimisation
)

A_est, B_est, C_est, D_est = result.x     # unpack the estimated parameters
 
print("=== Estimated Parameters ===")
print(f"  A (inertia)  = {A_est:.6f}")    # inertia coefficient
print(f"  B (damping)  = {B_est:.6f}")    # damping coefficient
print(f"  C (stiffness)= {C_est:.6f}")    # stiffness coefficient
print(f"  D (torque K) = {D_est:.6f}")    # motor torque constant
print(f"\nOptimiser converged: {result.status}")   # convergence status from LM solver
print(f"Final cost (sum sq. residuals): {np.sum(result.fun**2):.6e}")  # goodness of fit

xs_fit = simulate(result.x)               # run ODE one final time with estimated params
 
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)  # two stacked subplots
 
axes[0].plot(t_data, theta_meas,       label="Measured x",      color="steelblue")  # flight data
axes[0].plot(t_data, xs_fit[0, :], label="Simulated x",     color="tomato", linestyle="--")  # model
axes[0].set_ylabel("Aileron angle [rad]")   # y-axis label for top subplot
axes[0].legend()                            # show legend
axes[0].grid(True, alpha=0.3)              # light grid
 
axes[1].plot(t_data, theta_dot_meas,    label="Measured x_dot",  color="steelblue")  # flight data
axes[1].plot(t_data, xs_fit[1, :], label="Simulated x_dot", color="tomato", linestyle="--")  # model
axes[1].set_ylabel("Angular rate [rad/s]")  # y-axis label for bottom subplot
axes[1].set_xlabel("Time [s]")             # shared x-axis label
axes[1].legend()                            # show legend
axes[1].grid(True, alpha=0.3)              # light grid
 
plt.suptitle("Aileron Model: Simulated vs Measured")  # overall figure title
plt.tight_layout()                          # prevent label clipping
plt.savefig("aileron_fit.png", dpi=150)    # save figure to disk
plt.show()     