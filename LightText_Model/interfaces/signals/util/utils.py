import numpy as np
import matplotlib.pyplot as plt

def plot_frequency_spectrum(signal, fs):
    """
    Plots the frequency spectrum of the given signal.
    
    Parameters:
    - signal: The input signal (1D array)
    - fs: Sampling frequency of the signal in Hz
    """
    # Number of samples in the signal
    N = len(signal)
    
    # Apply FFT to the signal
    fft_values = np.fft.fft(signal)
    
    # Calculate the frequency axis (only positive frequencies)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Take the magnitude of the FFT values
    fft_magnitude = np.abs(fft_values)[:N//2]
    
    # Take the corresponding positive frequencies
    freqs = freqs[:N//2]
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()
