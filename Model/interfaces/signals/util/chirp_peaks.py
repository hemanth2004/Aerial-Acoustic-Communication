import numpy as np
import matplotlib.pyplot as plt

def generate_chirp_peaks_signal(num_peaks, peak_width, sample_rate, base_freq, max_freq, plot=False):
    """
    Generate a signal where the frequency of a sine wave is modulated in a triangular pattern.
    
    Parameters:
    - num_peaks: int, the number of frequency-modulated peaks in the signal.
    - peak_width: float, the time difference between the tips of two peaks in seconds.
    - sample_rate: int, the sampling rate in Hz.
    - base_freq: float, the minimum frequency in Hz at the base of the peaks.
    - max_freq: float, the maximum frequency in Hz at the peaks.
    
    Returns:
    - fm_signal: numpy array, the frequency-modulated signal.
    """
    
    # Total signal duration based on peak width and number of peaks
    signal_duration = num_peaks * peak_width
    
    # Total number of samples in the signal
    num_samples = int(signal_duration * sample_rate)
    
    # Time vector for the entire signal
    t = np.linspace(0, signal_duration, num_samples)

    # Create the triangular wave that will modulate the frequency
    # This triangular wave oscillates between -1 and 1
    triangle_wave = np.abs(2 * (t / peak_width % 1) - 1) * 2 - 1  # Create the triangular wave

    # Map the triangular wave to frequency values between base_freq and max_freq
    modulated_freq = base_freq + (max_freq - base_freq) * (triangle_wave + 1) / 2

    # Integrate the modulated frequency to get the phase
    phase = np.cumsum(2 * np.pi * modulated_freq / sample_rate)

    # Generate the FM signal as a sine wave using the modulated phase
    fm_signal = np.sin(phase)

    # Plot the modulated signal and frequency variation
    if plot:
        plt.figure(figsize=(10, 6))

        # Plot the frequency modulation
        plt.subplot(2, 1, 1)
        plt.plot(t, modulated_freq)
        plt.title(f'Frequency Modulation (Base Frequency = {base_freq} Hz, Max Frequency = {max_freq} Hz)')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.grid(True)

        # Plot the FM signal
        plt.subplot(2, 1, 2)
        plt.plot(t, fm_signal)
        plt.title('Frequency-Modulated Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return fm_signal