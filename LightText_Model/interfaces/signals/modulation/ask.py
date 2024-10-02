import numpy as np

high_amp = 1
low_amp = 0

def ask_modulation_order():
    return 2

def generate_ask_signal(bit_array, symbol_duration, sample_rate, carrier_freq=3700):
    samp_per_sym = int(symbol_duration * sample_rate)
    
    t = np.arange(samp_per_sym) / sample_rate
    ask_signal = np.array([])

    for bit in bit_array:
        if bit == 1:
            amp = high_amp
        else:
            amp = low_amp
        
        carrier_wave = amp * np.sin(2 * np.pi * carrier_freq * t)
        ask_signal = np.concatenate((ask_signal, carrier_wave))

    return ask_signal
        
def generate_ask_coding(bit_array, symbol_duration, sample_rate):
    samples_per_symbol = int(symbol_duration * sample_rate)
    signal = np.zeros(len(bit_array) * samples_per_symbol)

    for i, bit in enumerate(bit_array):
        start_index = i * samples_per_symbol
        if bit == 1:
            signal[start_index:start_index + samples_per_symbol] = 1  # High amplitude for bit 1
        else:
            signal[start_index:start_index + samples_per_symbol] = 0  # Low amplitude for bit 0

    return signal


def decode_ask_signal(signal, symbol_duration, sample_rate, threshold=-1):
    samp_per_sym = int(symbol_duration * sample_rate)
    
    bit_array = []

    if threshold < 0:
            threshold = (high_amp + low_amp) / 2

    for i in range(0, len(signal), samp_per_sym):
        symbol = signal[i:i+samp_per_sym]
        avg_amplitude = np.mean(np.abs(symbol))

        if avg_amplitude > threshold:
            bit_array.append(1)
        else:
            bit_array.append(0)
    
    return bit_array
