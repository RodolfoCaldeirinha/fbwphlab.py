import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter_butter import butter_filter
from signal_current_regression import fit_plane, derivative, remove_outliers

# CSV to numpy
def csv_to_numpy(file_path):
    df = pd.read_csv(file_path)
    
    data = df.to_numpy().flatten() # Ensure 1D array    
    return data

# Getting data
ticks = csv_to_numpy(r"Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/tick_subfile_4.csv")
ux = r"Master/data_toplevel/CSV data/simlog-20260218_130000/data/command/data/uxcmd_subfile_4.csv"
current_aileron = r"Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/IservoAil_subfile_4.csv"
delta_ail = r"Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/DeltaAil_subfile_4.csv"

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
current_aileron_np = csv_to_numpy(current_aileron)[:-1]

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