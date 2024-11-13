import numpy as np

freq = [700, 1000, 1300, 1600, 1900, 2200, 2500, 2800]


def fsk_modulation_order():
    return len(freq)


def generate_fsk_signal(bit_array, symbol_duration, sample_rate):
    samp_per_sym = int(symbol_duration * sample_rate)
    
    t = np.arange(samp_per_sym) / sample_rate
    fsk_signal = np.array([])

    for bit in bit_array:
        carrier_freq = freq[bit]
        carrier_wave = np.sin(2 * np.pi * carrier_freq * t)
        fsk_signal = np.concatenate((fsk_signal, carrier_wave))

    return fsk_signal
        
        
def decode_fsk_signal(signal, symbol_duration, sample_rate):
    samp_per_sym = int(symbol_duration * sample_rate)
    
    bit_array = []
    
    for i in range(0, len(signal), samp_per_sym):
        symbol = signal[i:i+samp_per_sym]
        freqs = np.fft.fftfreq(len(symbol), 1/sample_rate)
        fft_vals = np.abs(np.fft.fft(symbol))
        
        dominant_freq = freqs[np.argmax(fft_vals[:len(fft_vals)//2])]  # Get dominant frequency
        
        # Map the dominant frequency back to the bit
        closest_freq = min(freq, key=lambda x: abs(x - dominant_freq))
        bit = freq.index(closest_freq)
        
        bit_array.append(bit)
    
    return bit_array

