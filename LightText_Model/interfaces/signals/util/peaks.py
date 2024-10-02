import numpy as np
import matplotlib.pyplot as plt

def generate_peaks_signal(num_peaks, peak_width, sample_rate, carrier_freq, plot=False):
    """
    Generate a peaks signal composed of sharp peaks modulating a sine wave.
    
    Parameters:
    - num_peaks: int, the number of peaks in the signal.
    - peak_width: float, the time difference between the tips of two peaks in seconds.
    - sample_rate: int, the sampling rate in Hz.
    - carrier_freq: float, the frequency of the base sine wave in Hz.
    
    Returns:
    - modulated_signal: numpy array, the sharp-peaked modulated sine wave signal.
    """

    # Total signal duration based on peak width and number of peaks
    signal_duration = num_peaks * peak_width

    # Total number of samples in the signal
    num_samples = int(signal_duration * sample_rate)
    
    # Time vector for the entire signal
    t = np.linspace(0, signal_duration, num_samples)

    # Initialize the envelope with sharp peaks
    envelope = np.zeros(num_samples)

    # Calculate the positions of the peaks in the time vector
    peak_positions = np.linspace(peak_width / 2, signal_duration - peak_width / 2, num_peaks)

    # Generate sharp peaks using a triangular shape
    for peak_pos in peak_positions:
        # Calculate the index of the peak
        peak_idx = int(peak_pos * sample_rate)

        # Create a triangular (pointed) peak around the peak position
        peak_region = int(peak_width * sample_rate / 2)
        left_idx = max(0, peak_idx - peak_region)
        right_idx = min(num_samples, peak_idx + peak_region)

        # Create a triangular peak shape
        peak_shape = np.linspace(0, 1, peak_region)  # Rising part
        envelope[left_idx:peak_idx] = peak_shape
        envelope[peak_idx:right_idx] = peak_shape[::-1]  # Falling part

    # Generate the base sine wave (carrier signal)
    carrier_signal = np.sin(2 * np.pi * carrier_freq * t)

    # Modulate the sine wave with the sharp envelope
    modulated_signal = envelope * carrier_signal

    # Plot the modulated signal
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(t, modulated_signal)
        plt.title(f'Sharp Peak-modulated Sine Wave (Carrier Frequency = {carrier_freq} Hz)')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    return modulated_signal