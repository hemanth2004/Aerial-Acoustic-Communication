import numpy as np
import matplotlib.pyplot as plt

BOK_RANGE = (4000, 6000)

QOK_SWEEP1 = (1000, 2000) 
QOK_SWEEP2 = (2000, 3000) 
QOK_PARTITION = 0.7

def generate_css_bok_signal(bit_array, symbol_duration, sample_rate):
    samp_per_sym = int(symbol_duration * sample_rate)
    t = np.arange(samp_per_sym) / sample_rate
    signal = np.array([])

    for bit in bit_array:
        if bit == 1:
            freq = np.linspace(BOK_RANGE[0], BOK_RANGE[1], samp_per_sym)
        else:
            freq = np.linspace(BOK_RANGE[1], BOK_RANGE[0], samp_per_sym)

        chirp_signal = np.cos(2 * np.pi * freq * t)
        signal = np.concatenate((signal, chirp_signal))


    return signal

def generate_css_qok_signal(bit_array, symbol_duration, sample_rate):
    samp_per_sym = int(symbol_duration * sample_rate)
    t = np.arange(samp_per_sym) / sample_rate
    signal = np.array([])

    if len(bit_array) % 2 != 0:
        print("ERROR: bit array not even")
        return signal
    
    for bit1, bit2 in zip(bit_array[0::2], bit_array[1::2]):
        if bit1 == 0 and bit2 == 0:
            freq = np.linspace(QOK_SWEEP1[1], QOK_SWEEP1[0], samp_per_sym)
        elif bit1 == 0 and bit2 == 1:
            freq = np.linspace(QOK_SWEEP2[0], QOK_SWEEP2[1], samp_per_sym)
        elif bit1 == 1 and bit2 == 0:
            freq1 = np.linspace(QOK_SWEEP1[0], QOK_SWEEP1[1], int(QOK_PARTITION * samp_per_sym))
            freq2 = np.linspace(QOK_SWEEP1[1], QOK_SWEEP2[0], samp_per_sym - len(freq1))
            freq = np.concatenate((freq1, freq2))
        elif bit1 == 1 and bit2 == 1:
            freq1 = np.linspace(QOK_SWEEP2[1], QOK_SWEEP2[0], int(QOK_PARTITION * samp_per_sym))
            freq2 = np.linspace(QOK_SWEEP2[0], QOK_SWEEP1[1], samp_per_sym - len(freq1))
            freq = np.concatenate((freq1, freq2))

        chirp_signal = np.cos(2 * np.pi * freq * t)
        signal = np.concatenate((signal, chirp_signal))

    return signal


def decode_css_bok_signal(signal, symbol_duration, sampling_rate):
    # Accept a signal and decode it into a bit array. 
    # Signal is an array of samples
    # Number of symbols may vary based on the total signal duration.
    # Also return a list "borders" which contains all the indices of sample points where 
    # a symbol might have ended/started
    # So return both bit array and borders list.
    """
    Decode the BOK signal into a bit array and identify the symbol boundaries.
    
    Parameters:
    - signal: The received signal (1D array).
    - symbol_duration: The duration of each symbol in seconds.
    - sampling_rate: The sampling frequency of the signal in Hz.

    Returns:
    - bit_array: Decoded bit array.
    - borders: List of indices marking the start and end of each symbol.
    """

    samp_per_sym = int(symbol_duration * sampling_rate)
    num_symbols = len(signal) // samp_per_sym  # Number of symbols based on the total signal length
    bit_array = []
    borders = []

    # Define frequency thresholds based on BOK range
    bok_freq_low = BOK_RANGE[0]
    bok_freq_high = BOK_RANGE[1]

    for i in range(num_symbols):
        # Extract the segment of the signal corresponding to the current symbol
        start_index = i * samp_per_sym
        end_index = start_index + samp_per_sym
        segment = signal[start_index:end_index]

        # Perform FFT to find the dominant frequency in the segment
        fft_values = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1/sampling_rate)

        # Find the peak frequency
        peak_index = np.argmax(np.abs(fft_values[:len(segment)//2]))  # Only consider positive frequencies
        peak_freq = freqs[peak_index]

        # Check which bit the frequency corresponds to
        if bok_freq_low <= peak_freq <= bok_freq_high:
            bit_array.append(1)
        elif bok_freq_high < peak_freq <= (bok_freq_low + bok_freq_high) / 2:
            bit_array.append(0)
        else:
            # If the frequency is not in expected range, append a default value (could be an error state)
            # bit_array.append(-1)  # Indicate an error or unknown
            bit_array

        # Add to borders list
        borders.append(start_index)

    # Append the last border for the end of the signal
    borders.append(len(signal))

    return bit_array, borders

    return 