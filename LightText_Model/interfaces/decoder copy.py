import listener2 
import numpy as np
import os

import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import scipy.signal as signal
from signals.modulation.ask import generate_ask_signal, decode_ask_signal 
from signals.util.chirp_peaks import generate_chirp_peaks_signal
# ^ accepts signal, symbol duration and sample rate and returns a bit array
from encoding._5bit import char_from_bin
# ^ accepts a number from 0-31 and returns char

def plottt():
    """
    Plots the frequency spectrum of the given signal.
    
    Parameters:
    - signal: The input signal (1D array).
    - sampling_rate: The sampling frequency of the signal in Hz.
    """

    signal = np.array(sample_buffer)
    sampling_rate = 44100
    # Number of samples in the signal
    N = len(signal)
    
    # Apply FFT to the signal
    fft_values = np.fft.fft(signal)
    
    # Calculate the frequency axis (only positive frequencies)
    freqs = np.fft.fftfreq(N, 1/sampling_rate)
    
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


def print_buffer(bit_queue):
    temp_list = []
    
    while bit_queue:
        bit = bit_queue.popleft()  # Remove each bit from the queue
        temp_list.append(bit)  # Store it in a temporary list
        print(bit, end=" ")    
    
    print()
    for bit in temp_list:
        bit_queue.append(bit)

sample_buffer = deque() # Queue to store samples from listener's callback
bit_buffer = deque() # Queue to store decoded samples from bit_array
rate = 44100
symbol_duration = 0.25

preamble1 = 21 
preamble_bits = [1, 0, 1, 0, 1]  # The preamble we're looking for
preamble_signal = generate_ask_signal(np.array([preamble1]), 
                                      symbol_duration=symbol_duration, 
                                      sample_rate=rate)

# ^ two preambles and character encoding scheme must be followed

# Process Params
ideal_preamble = generate_chirp_peaks_signal(
    num_peaks=3, 
    peak_width=0.1, 
    sample_rate=44100, 
    base_freq=150, 
    max_freq=1500
)
preamble_len = 5
samples_per_symbol = int(symbol_duration * rate)
preamble_detected = False
frame_size = 5  # For 5-bit encoding

def find_preamble_end_in_buffer(bit_deque, preamble_bits):
    window_size = len(preamble_bits)
    bit_list = list(bit_deque)

    if len(bit_list) > window_size:
        for i in range(0, len(bit_list) - window_size):
            window =  bit_list[i:i + window_size]

            if (len(bit_list) - i) <= window_size + 1:
                print("Sliding Window:  ", window)

            if window == preamble_bits:
                return i
    
    return -1


def process_samples():
    """Process samples from the buffer to detect the preamble and decode bits."""
    global preamble_detected, preamble_bits, data_bits

    samples = []
    while sample_buffer:
        samples.append(sample_buffer.popleft())

    
    if len(samples) < samples_per_symbol:
        return 

    bits = decode_ask_signal(np.array(samples), symbol_duration, rate, threshold=0.01)
    for i in range(len(bits)):
        bit_buffer.append(bits[i])
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print()
    preamble_index = find_preamble_end_in_buffer(bit_buffer, preamble_bits)
    print("Preamble is in buffer index", preamble_index)
    
    # if preamble_index != -1:
    #         for i in range(preamble_index):
    #             bit_buffer.pop()

    print("bits: ", end="")
    print_buffer(bit_buffer)

def detect_preamble(buffer, ideal_preamble, sample_rate, threshold=None):
    """
    Detects the location of the ideal preamble in a buffer using cross-correlation.
    
    Parameters:
    - buffer: numpy array, the recorded signal of arbitrary length.
    - ideal_preamble: numpy array, the known preamble signal.
    - sample_rate: int, the sample rate in Hz.
    - threshold: float, optional threshold to filter low correlation peaks. If None, the peak correlation is used.
    
    Returns:
    - preamble_start: int, the index of the start of the preamble in the buffer.
    - preamble_end: int, the index of the end of the preamble in the buffer.
    - correlation: numpy array, the cross-correlation result.
    """
    
    # Perform cross-correlation between the buffer and the ideal preamble
    correlation = signal.correlate(buffer, ideal_preamble, mode='valid')
    
    # If a threshold is provided, filter out low correlation peaks
    if threshold is not None:
        correlation[correlation < threshold] = 0
    
    # Find the index of the maximum correlation value
    preamble_start = np.argmax(correlation)
    preamble_end = preamble_start + len(ideal_preamble)

    return preamble_start, preamble_end, correlation



def update_callback(data):
    """Callback function to process data."""

    for a in data:
        sample_buffer.append(a)
    
    # print(np.max(np.abs(data)))
    process_samples()
    # _1, _2, correlation = detect_preamble(
    #     buffer=np.reshape(np.array(sample_buffer), (-1,)), 
    #     sample_rate=rate, 
    #     ideal_preamble=ideal_preamble
    #     )
    
    # print("Preamble correlation = ", np.max(correlation))

    # if len(bit_buffer) > 40:
    #     bit_buffer.popleft()

from collections import deque

listener2.decoder_callbacks.append(update_callback)
listener2.start_listening()