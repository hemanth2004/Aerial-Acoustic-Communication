import numpy as np
from scipy.signal import correlate
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import time

def load_signal(file_name):
    """
    Load a WAV file and return the signal array and sampling rate.

    Parameters:
    - file_name (str): The path to the WAV file to load.

    Returns:
    - tuple: (sampling_rate, signal) where sampling_rate is the sample rate (in Hz)
             and signal is a NumPy array of the audio samples.
    """
    try:
        # Read the WAV file
        sampling_rate, signal = read(file_name)
        print(f"Loaded '{file_name}' with sampling rate {sampling_rate} Hz")

        return sampling_rate, signal
    except Exception as e:
        print(f"Error loading file '{file_name}': {e}")
        return None, None

from signals.modulation.css import generate_css_bok_signal


def calculate_bok_correlation_matrix_v2(upchirp_recording, downchirp_recording, upchirp_reference, downchirp_reference):
    """
    Calculate the BOK correlation matrix using separate upchirp and downchirp recordings.

    Parameters:
    - upchirp_recording (np.array): The recorded signal segment corresponding to an upchirp.
    - downchirp_recording (np.array): The recorded signal segment corresponding to a downchirp.
    - upchirp_reference (np.array): Reference upchirp signal.
    - downchirp_reference (np.array): Reference downchirp signal.

    Returns:
    - np.array: A 2x2 correlation matrix, where:
        [0, 0]: correlation of upchirp recording with upchirp reference
        [0, 1]: correlation of upchirp recording with downchirp reference
        [1, 0]: correlation of downchirp recording with upchirp reference
        [1, 1]: correlation of downchirp recording with downchirp reference
    """
    correlation_matrix = np.zeros((2, 2))  # Initialize a 2x2 matrix for BOK

    # Calculate correlations of upchirp recording with both references
    correlation_matrix[0, 0] = np.max(correlate(upchirp_recording, upchirp_reference, mode='valid'))
    correlation_matrix[0, 1] = np.max(correlate(upchirp_recording, downchirp_reference, mode='valid'))

    # Calculate correlations of downchirp recording with both references
    correlation_matrix[1, 0] = np.max(correlate(downchirp_recording, upchirp_reference, mode='valid'))
    correlation_matrix[1, 1] = np.max(correlate(downchirp_recording, downchirp_reference, mode='valid'))

    return correlation_matrix

def play_and_record(signal, fs, record_duration, output_file):
    """
    Play a signal through speakers, record the microphone input for a set duration, and save to a WAV file.

    Parameters:
    - signal (np.array): The signal to be played.
    - fs (int): Sampling frequency in Hz.
    - record_duration (float): Duration of recording in seconds.
    - output_file (str): Path to save the recorded file.
    """
    # Ensure signal duration matches the record_duration
    desired_samples = int(record_duration * fs)
    if len(signal) > desired_samples:
        signal = signal[:desired_samples]
    elif len(signal) < desired_samples:
        signal = np.pad(signal, (0, desired_samples - len(signal)), 'constant')

    try:
        # Start playback and record simultaneously
        print("Starting playback and recording...")
        recorded_audio = sd.playrec(signal, samplerate=fs, channels=1, dtype='float32')
        
        # Wait for playback and recording to complete
        sd.wait()
        print("Recording finished.")

        # Save the recorded audio to a WAV file
        write(output_file, fs, recorded_audio)
        print(f"Recording saved as '{output_file}'")


    except Exception as e:
        print(f"Error during playback and recording: {e}")

def save_signal(signal, fs, output_file):
    """
    Save a given signal directly to a WAV file.

    Parameters:
    - signal (np.array): The signal to be saved.
    - fs (int): Sampling frequency in Hz.
    - output_file (str): Path to save the file as a WAV.
    """
    # Normalize the signal to be in the range of int16 if required for WAV format
    # Convert signal to float32 and ensure it is in the range [-1, 1] for proper WAV file saving
    if signal.dtype != np.float32:
        signal = np.array(signal, dtype=np.float32)
        max_val = np.max(np.abs(signal))
        if max_val > 0:  # Avoid division by zero
            signal /= max_val  # Normalize to [-1, 1]
    
    # Save the signal to a WAV file
    try:
        write(output_file, fs, signal)
        print(f"Signal saved as '{output_file}'")
    except Exception as e:
        print(f"Error saving signal: {e}")


# Parameters
fs = 44100                # Sampling frequency
record_duration = 0.25   # Duration to record (same as playback duration)
output_file = "recorded_downchirp.wav"  # Output filename for the recording

# Generate the chirp signal
upchirp = generate_css_bok_signal([1], record_duration, fs)
downchirp = generate_css_bok_signal([0], record_duration, fs)

target_chirp = upchirp

# Play the chirp and record the response
#play_and_record(upchirp, fs, record_duration, "upchirp.wav")
#time.sleep(0.25)
#play_and_record(downchirp, fs, record_duration, "downchirp.wav")
#time.sleep(0.25)
recorded_upchirp = load_signal("upchirp.wav")[1]
recorder_downchirp = load_signal("downchirp.wav")[1]

confusion_matrix = calculate_bok_correlation_matrix_v2(recorded_upchirp, recorder_downchirp, upchirp, downchirp)
print(confusion_matrix)


import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(correlation_matrix):
    """
    Plot a confusion matrix using a heatmap.

    Parameters:
    - correlation_matrix (np.array): The 2x2 correlation matrix to plot.
    """
    # Create a heatmap for the correlation matrix
    plt.figure(figsize=(3, 3))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Upchirp Ref.', 'Downchirp Ref.'], 
                yticklabels=['Upchirp Rec.', 'Downchirp Rec.'], 
                cbar_kws={'label': 'Correlation Value'})

    plt.title('BOK Correlation Matrix')
    plt.xlabel('Reference Chirp')
    plt.ylabel('Recorded Chirp')
    plt.show()

#
# Example usage (assuming the confusion matrix is stored in `confusion_matrix`):
plot_correlation_matrix(confusion_matrix)

