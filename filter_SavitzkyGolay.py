import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def convert_array(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [float(line.strip()) for line in lines[1:]]  # skip header, convert all

fs = 10000   #sampling frequency
t_begin = 982.7500 #[s]
t_end = 1910.1990 #[s]

x = convert_array("Master/Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/Axb.csv")
t = np.linspace(t_begin, t_end, len(x))

# Apply Savitzky–Golay filter
x_smooth = savgol_filter(x, window_length=65, polyorder=2)

# Plot
plt.plot(t, x, label="Noisy signal", alpha=0.5)
plt.plot(t, x_smooth, label="Smoothed signal", linewidth=2)
plt.legend()
plt.show()
print("a")