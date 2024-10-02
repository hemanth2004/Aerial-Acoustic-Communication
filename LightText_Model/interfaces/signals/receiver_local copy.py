import numpy as np
import wave
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate

# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 0.05      # Duration of each symbol in seconds
frequencies = [300, 600, 900, 1200]  # 4 FSK frequencies
amplitude_levels = [0.1, 0.33, 0.66, 1.0]  # 4 ASK amplitude levels

# Read the WAV file
def read_wav(filename):
    with wave.open(filename, 'rb') as wf:
        params = wf.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes
        audio_data = wf.readframes(n_frames)
        signal = np.frombuffer(audio_data, dtype=np.int16)
    return signal, framerate

# Decode the FSK-ASK signal
def decode_fsk_ask_signal(signal, sample_rate, duration):
    symbol_duration = int(sample_rate * duration)
    num_samples = len(signal)
    num_symbols = num_samples // symbol_duration
    decoded_symbols = []
    
    t = np.linspace(0, duration, symbol_duration, endpoint=False)
    
    print(f"Symbol Duration: {symbol_duration}")
    print(f"Number of Samples: {num_samples}")
    print(f"Number of Symbols Detected: {num_symbols}")
    
    for i in range(num_symbols):
        start = i * symbol_duration
        end = start + symbol_duration
        segment = signal[start:end]
        
        # Normalize the segment
        segment = segment / np.max(np.abs(segment))
        
        # Initialize variables to detect the closest frequency and amplitude
        best_frequency = None
        max_correlation = -float('inf')
        
        for freq in frequencies:
            # Generate a reference sinusoid for the current frequency
            reference_wave = np.sin(2 * np.pi * freq * t)
            
            # Calculate the correlation between the signal segment and the reference wave
            correlation = np.sum(segment * reference_wave) / (np.sqrt(np.sum(segment ** 2)) * np.sqrt(np.sum(reference_wave ** 2)))
            
            if correlation > max_correlation:
                max_correlation = correlation
                best_frequency = freq
        
        # Estimate amplitude using the RMS value
        estimated_amplitude = np.sqrt(np.mean(segment ** 2))
        closest_amplitude = min(amplitude_levels, key=lambda x: abs(x - estimated_amplitude))
        
        # Convert frequency and amplitude to a symbol
        freq_index = frequencies.index(best_frequency)
        amp_index = amplitude_levels.index(closest_amplitude)
        symbol = freq_index * len(amplitude_levels) + amp_index
        
        decoded_symbols.append(symbol)
    
    return decoded_symbols

# Convert symbols to binary arrays
def symbols_to_binary(symbols):
    return [format(symbol, '06b') for symbol in symbols]

# Main
def main():
    # Read the signal from the WAV file
    signal, sample_rate = read_wav('fsk_ask_signal.wav')
    
    # Decode the signal
    decoded_symbols = decode_fsk_ask_signal(signal, sample_rate, duration)
    
    if not decoded_symbols:
        print("No symbols decoded.")
    else:
        # Convert to binary arrays
        binary_arrays = symbols_to_binary(decoded_symbols)
        print("Decoded 6-bit symbols:")
        df = pd.DataFrame(binary_arrays)
        print(df)
        print(decoded_symbols)
    
    # Optionally, plot the waveform for verification
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4)) 
    plt.plot(t, signal)
    plt.title('Waveform of the Received Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
