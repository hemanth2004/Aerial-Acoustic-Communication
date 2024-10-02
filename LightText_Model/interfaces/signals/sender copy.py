import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import wave

# Parameters
sample_rate = 44100  # Sample rate in Hz
duration = 0.05      # Duration of each symbol in seconds
frequencies = [300, 600, 900, 1200]  # 4 FSK frequencies
amplitude_levels = [0.1, 0.33, 0.66, 1.0]  # 4 ASK amplitude levels

# Generate FSK-ASK signal
def generate_fsk_ask_signal(data, duration, sample_rate):
    samples_per_symbol = int(sample_rate * duration)
    t = np.linspace(0, duration, samples_per_symbol, endpoint=False)
    signal = np.array([])

    for symbol in data:
        frequency = frequencies[symbol // len(amplitude_levels)]  # Higher bits for FSK
        amplitude = amplitude_levels[symbol % len(amplitude_levels)]  # Lower bits for ASK
        waveform = amplitude * np.sin(2 * np.pi * frequency * t)
        signal = np.concatenate((signal, waveform))

    return signal

# Save the signal to a WAV file
def save_to_wav(signal, sample_rate, filename):
    signal = np.int16(signal / np.max(np.abs(signal)) * 32767)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

# Emit the signal via speaker
def play_signal(signal, sample_rate):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)
    stream.write(signal.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def send_bits(bit_array):
    # Generate the FSK-ASK signal for the data
    signal = generate_fsk_ask_signal(bit_array, duration, sample_rate)
    print("Total samples: ", len(signal))
    
    # Play the generated signal
    play_signal(signal, sample_rate)

    # Save the generated signal to a WAV file
    filename = 'fsk_ask_signal.wav'
    save_to_wav(signal, sample_rate, filename)
    
    # Optionally, plot the waveform
    t = np.arange(len(signal)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal)
    plt.title('Waveform of the Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
