import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, sosfilt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Function to design and apply the Butterworth bandpass filter
def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    # Design Butterworth bandpass filter
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter to the signal
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def narrowband_filter(samples, center_freq, bandwidth, sample_rate, order=4):
    """
    Apply a narrowband Butterworth filter to isolate a specific frequency.

    Parameters:
    samples (numpy array): The input signal to be filtered.
    center_freq (float): The center frequency to isolate (in Hz).
    bandwidth (float): The bandwidth around the center frequency (in Hz).
    sample_rate (float): The sampling rate of the input signal (in Hz).
    order (int): The order of the Butterworth filter.

    Returns:
    numpy array: The filtered signal.
    """
    # Calculate the lower and upper cutoff frequencies based on the center and bandwidth
    nyquist_rate = sample_rate / 2.0
    low_cutoff = (center_freq - bandwidth / 2) / nyquist_rate
    high_cutoff = (center_freq + bandwidth / 2) / nyquist_rate

    # Design a Butterworth bandpass filter using the second-order sections (sos) format
    sos = butter(order, [low_cutoff, high_cutoff], btype='band', output='sos')

    # Apply the filter to the input signal
    filtered_signal = sosfilt(sos, samples)

    return filtered_signal
