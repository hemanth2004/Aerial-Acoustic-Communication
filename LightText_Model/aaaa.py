import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from interfaces.signals.modulation.css import generate_css_bok_signal

def frequency_time_filter(signal, sample_rate):
    """
    Converts the given signal into a frequency-time representation.
    This function calculates the instantaneous frequency using the Hilbert transform.
    
    Parameters:
    - signal (numpy array): Input chirp signal.
    - sample_rate (float): The sampling rate of the signal.
    
    Returns:
    - time (numpy array): Time axis.
    - inst_freq (numpy array): Instantaneous frequency at each time point.
    """
    # Compute analytic signal using Hilbert transform
    analytic_signal = hilbert(signal)
    
    # Extract the instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic_signal))
    
    # Compute the instantaneous frequency as the derivative of the phase
    inst_freq = np.diff(inst_phase) * (sample_rate / (2.0 * np.pi))
    
    # Time axis for plotting (excluding the last point since freq has one less element)
    time = np.arange(len(inst_freq)) / sample_rate
    
    return time, inst_freq

time_axis, inst_freq = frequency_time_filter(generate_css_bok_signal([1,0,1,0,1,0], 0.1, 44100), 44100)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(time_axis, inst_freq, label="Instantaneous Frequency", color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Frequency-Time Representation of the Upchirp Signal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
