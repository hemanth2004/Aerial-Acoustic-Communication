from scipy.signal import savgol_filter

def savitzky_golay_filter(signal, polyorder):
    window_size = min(len(signal), 51)  # Use the smaller of signal length or default window size
    if window_size % 2 == 0:
        window_size += 1  # Ensure the window size is odd
    
    # Apply Savitzky-Golay filter if the window size is valid, else return original signal
    if window_size > polyorder:
        return savgol_filter(signal, window_size, polyorder)
    else:
        return signal  # If signal is too short, don't apply the filter
