import numpy as np
import pandas as pd
from get_data import data_grabber
from sklearn.linear_model import LinearRegression
# As a reminder
# ux = roll = aileron
# uy = pitch = elevator
# CSV to numpy
def csv_to_numpy(file_path):
    df = pd.read_csv(file_path)
    
    data = df.to_numpy().flatten() # Ensure 1D array    
    return data

ux, ticks = data_grabber("simlog-20260218_130000","command","uxcmd_subfile_4", True)
ux = ux.to_numpy().flatten()
ticks = ticks.to_numpy().flatten()
Iail = data_grabber("simlog-20260218_130000","aircraft","IservoAil_subfile_4", False).to_numpy().flatten()
delta_ail = data_grabber("simlog-20260218_130000","aircraft","DeltaAil_subfile_4", False).to_numpy().flatten()
#ticks = csv_to_numpy(r'Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/tick_subfile_4.csv')
#ux = csv_to_numpy(r'Master/data_toplevel/CSV data/simlog-20260218_130000/data/command/data/uxcmd_subfile_4.csv')
#Iail = csv_to_numpy(r'Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/IservoAil_subfile_4.csv')
#delta_ail = csv_to_numpy(r'Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/DeltaAil_subfile_4.csv')

#print(len(ticks))
#print(len(ux))
#print(len(Iail))
#print(len(delta_ail))
#print(np.max(Iail))

import numpy as np

def derivative(x, t):
    #if x is not np.ndarray:
    #    x = csv_to_numpy(x)
    dt = np.diff(t)
    dx = np.diff(x)
    return dx/dt

# Taking the derivative of the aileron deflection to get the aileron rate
ddeltadot_ail = derivative(delta_ail, ticks)

ux    = ux[:-1]     # trim to n-1
Iail  = Iail[:-1]   # trim to n-1
ticks = ticks[:-1]  # trim to n-1

#print(len(ddeltadot_ail))
#print(np.max(ux))
#print(np.min(Iail))

# Vector of data
X = np.column_stack((ux, ddeltadot_ail))

def remove_outliers(X, y, n):
    """
    Remove the top and bottom N outliers based on y values.
    Returns filtered X and y with the same row alignment.
    """
    lower = np.partition(y, n)[n]           # nth smallest
    upper = np.partition(y, -n - 1)[-n - 1] # nth largest
    mask = (y >= lower) & (y <= upper)
    return X[mask], y[mask]

def fit_plane(X, y, n_outliers):
    # Remove outliers
    X, y = remove_outliers(X, y, n_outliers)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X, y)
    
    return model