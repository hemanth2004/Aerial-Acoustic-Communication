import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import wave

sample_rate = 44100  # Sample rate in Hz
duration = 0.075  # Duration of each symbol in seconds

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

def send_bits(bit_array, modulation_fn, plot_wave=True, preamble_signal=None):
    # Generate the FSK-ASK signal for the data
    signal = modulation_fn(bit_array, duration, sample_rate)
    if preamble_signal is not None:
        signal = np.concatenate((preamble_signal, signal))

    print("Total samples: ", len(signal))
    
    # Play the generated signal
    play_signal(signal, sample_rate)

    # Save the generated signal to a WAV file
    filename = 'ask_signal.wav'
    save_to_wav(signal, sample_rate, filename)
    
    # Optionally, plot the waveform
    if plot_wave:
        t = np.arange(len(signal)) / sample_rate
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal)
        plt.title('Waveform of the Signal')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
