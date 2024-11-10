import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, correlate, convolve
from scipy.signal import butter, lfilter, filtfilt, sosfilt, stft

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def l_butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sos_butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to data.
    
    Parameters:
        data (array-like): Input signal to filter.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): The order of the filter.
        
    Returns:
        filtered_data (ndarray): Bandpass-filtered signal.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order)
    filtered_data = sosfilt(sos, data)
    return filtered_data

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


def mag2freq_filter(signal, sample_rate, window='hann', nperseg=256, noverlap=None):
    """
    Converts a magnitude-time signal to a frequency-time signal using STFT.

    Parameters:
    - signal (numpy array): The input magnitude-time signal.
    - sample_rate (float): The sample rate of the signal (in Hz).
    - window (str or tuple or array_like, optional): Desired window to use (default is 'hann').
    - nperseg (int, optional): Length of each segment (default is 256).
    - noverlap (int, optional): Number of points to overlap between segments (default is None).

    Returns:
    - f (numpy array): Array of sample frequencies.
    - t (numpy array): Array of segment times.
    - Zxx (2D numpy array): STFT of the signal, representing the frequency-time signal.
    """
    # Compute the STFT of the signal
    f, t, Zxx = stft(signal, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)

    return f, t, Zxx

def frequency_time_filter(signal, sample_rate, amplitude_threshold=0.01, freq_range=(0, 20000)):
    """
    Converts the given signal into a frequency-time representation, 
    handling noise and limiting the output to the specified frequency range.
    
    Parameters:
    - signal (numpy array): Input chirp signal.
    - sample_rate (float): The sampling rate of the signal.
    - amplitude_threshold (float): Minimum amplitude to consider for frequency detection.
    - freq_range (tuple): (min_freq, max_freq) range to limit the frequency output.
    
    Returns:
    - time (numpy array): Time axis.
    - inst_freq (numpy array): Instantaneous frequency at each time point.
    """
    min_freq, max_freq = freq_range
    
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)
    
    # Get amplitude and phase of the analytic signal
    amplitude_envelope = np.abs(analytic_signal)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    
    # Apply amplitude thresholding (ignore frequencies where amplitude is too low)
    valid_amplitude = amplitude_envelope > amplitude_threshold
    inst_phase = np.where(valid_amplitude, inst_phase, 0)
    
    # Compute instantaneous frequency as the derivative of phase
    inst_freq = np.diff(inst_phase) * (sample_rate / (2.0 * np.pi))
    
    # Clip frequencies to be within the specified frequency range
    inst_freq = np.clip(inst_freq, min_freq, max_freq)
    
    # Time axis for plotting (excluding the last point since freq has one less element)
    time = np.arange(len(inst_freq)) / sample_rate
    
    return time, inst_freq


def correlation_filter(signal, sample_rate, reference_signal):
    """
    Applies a correlation filter to the signal based on the given reference signal.

    Parameters:
    - signal (numpy array): The input signal to be filtered.
    - sample_rate (float): The sampling rate of the signal (in Hz).
    - reference_signal (numpy array): The reference signal (e.g., chirp) for correlation.

    Returns:
    - time (numpy array): Time axis.
    - correlated_signal (numpy array): Correlated signal (correlation strength over time).
    """
    # Perform correlation between the input signal and reference signal
    correlated_signal = correlate(signal, reference_signal, mode='same')

    # Normalize the correlation result
    correlated_signal = correlated_signal / np.max(np.abs(correlated_signal))

    # Generate time axis for plotting
    time = np.arange(len(correlated_signal)) / sample_rate

    return time, correlated_signal



def matched_filter(signal, sample_rate, reference_signal):
    """
    Applies a matched filter to the signal based on the given reference signal.

    Parameters:
    - signal (numpy array): The input signal to be filtered.
    - sample_rate (float): The sampling rate of the signal (in Hz).
    - reference_signal (numpy array): The reference signal (e.g., chirp) for matched filtering.

    Returns:
    - time (numpy array): Time axis.
    - matched_signal (numpy array): Matched filter output (matched filter response over time).
    """
    # Create the matched filter by conjugating and time-reversing the reference signal
    matched_filter_template = np.conj(reference_signal[::-1])

    # Perform convolution (matched filtering) between the input signal and the matched filter template
    matched_signal = convolve(signal, matched_filter_template, mode='same')

    # Normalize the matched filter output
    matched_signal = matched_signal / np.max(np.abs(matched_signal))

    # Generate time axis for plotting
    time = np.arange(len(matched_signal)) / sample_rate

    return time, matched_signal
