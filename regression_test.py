import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from get_data import data_grabber
from filter_butter import butter_filter
# from signal_current_regression import fit_plane, derivative, remove_outliers

'''RELEVANT FUNCTIONS'''
# CSV to numpy
def csv_to_numpy(file_path):
    df = pd.read_csv(file_path)
    
    data = df.to_numpy().flatten() # Ensure 1D array    
    return data

# Derivative function
def derivative(x, t):
    #if x is not np.ndarray:
    #    x = csv_to_numpy(x)
    dt = np.diff(t)
    dx = np.diff(x)
    return dx/dt

# Remove outliers function
def remove_outliers(X, y, n):
    """
    Remove the top and bottom N outliers based on y values.
    Returns filtered X and y with the same row alignment.
    """
    lower = np.partition(y, n)[n]           # nth smallest
    upper = np.partition(y, -n - 1)[-n - 1] # nth largest
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]

# Regression model
def fit_plane(X, y, n_outliers):
    # Remove outliers
    X, y = remove_outliers(X, y, n_outliers)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X, y)
    
    return model

'''MAIN CODE: REGRESSION IN ACTION'''
# Getting data
ux, ticks = data_grabber("simlog-20260218_130000","command","uxcmd_subfile_4", True)
current_aileron = data_grabber("simlog-20260218_130000","aircraft","IservoAil_subfile_4", False)
delta_ail = data_grabber("simlog-20260218_130000","aircraft","DeltaAil_subfile_4", False)

#Rodolfo's Correction
ux = ux.to_numpy().flatten()
ticks = ticks.to_numpy().flatten()
current_aileron = current_aileron[0].to_numpy().flatten()
delta_ail = delta_ail[0].to_numpy().flatten()

# Filtering (commented out)
ux_np = butter_filter(ux, 1000, ticks[0], ticks[-1], filter_type='low', filtering_order=2, cut_off=15)
current_aileron_np = butter_filter(current_aileron, 1000, ticks[0], ticks[-1], filter_type='low', filtering_order=2, cut_off=15)
delta_ail_np = butter_filter(delta_ail, 1000, ticks[0], ticks[-1], filter_type='low', filtering_order=2, cut_off=15)

# Derivative of aileron deflection
#delta_ail_np = csv_to_numpy(delta_ail)
#ux_np = csv_to_numpy(ux)
ddeltadot_ail = derivative(delta_ail_np, ticks)

# np.diff shortens by 1 — trim all arrays to match
ux_np  = ux_np[:-1]
ticks  = ticks[:-1]
#Rodolfo's Fix
current_aileron_np = current_aileron_np[:-1]

X = np.column_stack((ux_np, ddeltadot_ail))

# Regression — remove top/bottom N outliers before fitting
N_OUTLIERS = 50  # adjust as needed
model = fit_plane(X, current_aileron_np, n_outliers=N_OUTLIERS)

# For prediction, use the cleaned data so X and y stay aligned
X_clean, current_clean = remove_outliers(X, current_aileron_np, n=N_OUTLIERS)
current_pred = model.predict(X_clean)

ticks = (ticks - ticks[0]) * 1e-4  # Convert ticks to seconds (1 tick = 0.0001 s)

# Plotting the result
ux_clean = X_clean[:, 0]  # voltage column after outlier removal
plt.plot(ux_clean, current_pred, 'b-', label='Linear Fit')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A]')
plt.legend()
plt.show()