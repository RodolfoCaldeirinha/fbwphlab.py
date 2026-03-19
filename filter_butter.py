import scipy
import numpy as np
import matplotlib.pyplot as plt

fs = 10000   #sampling frequency
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
    x = convert_array("Master/Master/data_toplevel/CSV data/simlog-20260218_130000/data/aircraft/data/Axb.csv")
    t = np.linspace(t_begin, t_end, len(x))
    b, a = scipy.signal.butter(N=filtering_order, Wn=cut_off, btype=filter_type, fs=fs) #(order, cutoff frequency, filter type(low or hgih), sampling frequency)

    x_filtered = scipy.signal.filtfilt(b, a, x)
    return x_filtered

print("a")
# plt.figure(figsize=(10, 4))
# plt.plot(t, x, alpha=0.5, label='Noisy signal')
# plt.plot(t, x_filtered, linewidth=2, label='Filtered (filtfilt, 0 lag)')
# plt.xlabel('Time (s)')
# plt.legend()
# plt.tight_layout()
# plt.show()