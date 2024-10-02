import numpy as np
import wave
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate

# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 0.05      # Duration of each symbol in seconds

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

# Convert symbols to binary arrays
def symbols_to_binary(symbols):
    return [format(symbol, '06b') for symbol in symbols]

def recorded_read(demodulation_fn, file_name):
    signal, sample_rate = read_wav('{fn}.wav'.format(fn=file_name))
    
    return demodulation_fn(signal, duration, sample_rate)
    
    if not decoded_symbols:
        print("No symbols decoded.")
    else:
        # Convert to binary arrays
        binary_arrays = symbols_to_binary(decoded_symbols)
        print("Decoded 6-bit symbols:")
        df = pd.DataFrame(binary_arrays)
        print(df)
        print(decoded_symbols)

        return binary_arrays
    
    # Optionally, plot the waveform for verification
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4)) 
    plt.plot(t, signal)
    plt.title('Waveform of the Received Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
