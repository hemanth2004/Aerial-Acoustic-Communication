import numpy as np
import matplotlib.pyplot as plt

BOK_RANGE = (3000, 5000)


QOK_SWEEP1 = (3000, 4000) 
QOK_SWEEP2 = (4000, 5000) 
QOK_RANGE = (QOK_SWEEP1[0], QOK_SWEEP2[1])
QOK_PARTITION = 0.7

CTS_RANGE = (4000, 6000)

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

def generate_css_cts_signal(bit_array, symbol_duration, sample_rate):
    samp_per_sym = int(symbol_duration * sample_rate)
    t = np.arange(samp_per_sym) / sample_rate
    signal = np.array([])

    mid_range = (CTS_RANGE[1] - CTS_RANGE[0]) / 2 + CTS_RANGE[0]  # Mid-range frequency for CTS

    for bit in bit_array:
        if bit == 1:  # Upchirp
            # First half: from mid-range to max (CTS_RANGE[1])
            freq1 = np.linspace(mid_range, CTS_RANGE[1], samp_per_sym // 2)
            # Second half: from min (CTS_RANGE[0]) to mid-range
            freq2 = np.linspace(CTS_RANGE[0], mid_range, samp_per_sym - len(freq1))
            freq = np.concatenate((freq1, freq2))
        else:  # Downchirp
            # First half: from mid-range to min (CTS_RANGE[0])
            freq1 = np.linspace(mid_range, CTS_RANGE[0], samp_per_sym // 2)
            # Second half: from max (CTS_RANGE[1]) to mid-range
            freq2 = np.linspace(CTS_RANGE[1], mid_range, samp_per_sym - len(freq1))
            freq = np.concatenate((freq1, freq2))

        chirp_signal = np.cos(2 * np.pi * freq * t)
        signal = np.concatenate((signal, chirp_signal))

    return signal
