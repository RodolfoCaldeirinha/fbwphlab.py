import scipy

def convert_array(file):
    with open(file, 'r') as f:
            lines = f.readlines()
    return [float(line.strip()) for line in lines[1:]]  # skip header, convert all

def butter_filter(x,  fs, t_begin, t_end, filter_type = 'low', filtering_order = 2, cut_off = 15,):
    b, a = scipy.signal.butter(N=filtering_order, Wn=cut_off, btype=filter_type, fs=fs) 
                            #(order, cutoff frequency, filter type(low or hgih), sampling frequency)
    x_filtered = scipy.signal.filtfilt(b, a, x) #Rodolfo's Fix 3 
    return x_filtered
