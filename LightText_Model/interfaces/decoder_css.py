import listener2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from collections import deque

from signals.util.rt import realtime_process
from signals.processing.filters import moving_average, frequency_time_filter, bandpass_filter, correlation_filter
from signals.modulation.css import generate_css_bok_signal, generate_css_qok_signal, generate_css_cts_signal
from signals.processing.synchronization import sync_ask
from signals.modulation.css import CTS_RANGE, BOK_RANGE, QOK_RANGE, QOK_SWEEP1
from frame_processing import get_frame

# engine on parameters
power_mean_threshold = 0.001 # for pre preamble phase
power_max_threshold = 0.01 # for pre and post preamble phase

rate = 44100
symbol_duration = 0.075
frame_size = 1 + 5 * 2

in_frame = False

fft_fig = None

#region FFT realtime process
def ftp_out(output):
    global fft_fig
    plt.close(fft_fig)

    # Calculate magnitude of the FFT output
    magnitude = np.abs(output)
    
    # Compute the frequency bins for the FFT output
    freq = np.fft.fftfreq(len(magnitude), d=1/ftp.sample_rate)

    # Only take the positive frequencies
    positive_freq_idx = np.where(freq >= 2)
    freq = freq[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]

    # Now filter out frequencies above 50 Hz
    valid_idx = np.where(freq <= 6000)  # Get indices where frequency <= 20 Hz
    filtered_freq = freq[valid_idx]
    filtered_magnitude = magnitude[valid_idx]

    fft_fig = plt.figure()
    plt.plot(filtered_freq, filtered_magnitude)
    plt.title('FFT Output (<= 20 Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
ftp = realtime_process(
    process=np.fft.fft, 
    outlet=ftp_out, 
    deadline=8, 
    sample_rate=44100
)
#endregion

#region Differentiate Envelope
def dif_out(samples):
    print(len(samples))
    global fft_fig
    if False:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        fft_fig = plt.figure()

        #Plot the envelope
        time_axis = np.arange(len(samples)) / 44100  # Create a time axis for plotting
        plt.plot(time_axis, samples, label='d/dx Envelope', color='orange')
        plt.title('d/dx Hilbert Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
dif = realtime_process(
    process=np.gradient,
    outlet=dif_out, 
    deadline=8,
    sample_rate=44100
)
#endregion


#region Mag2Freq Filter realtime process
def m2f_process(input):
    return moving_average(frequency_time_filter(input, 
                           sample_rate= 44100,
                           freq_range=BOK_RANGE)[1],
                           window_size=2100)

def m2f_out(output):
    global fft_fig
    f = output
    dif.update(f)
    if False:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        # Create a new figure for plotting
        fft_fig = plt.figure()
        time_axis = np.arange(len(f)) / 44100  # Create a time axis for plotting
        # Plot both the original and filtered signals for comparison
        plt.plot(time_axis, f, label='Filtered Signal', color='blue', alpha=0.5)
        plt.title('Mag2Freq Filter')
        plt.xlabel('')
        plt.ylabel('')
        plt.grid(True)
        plt.legend()
        plt.show()
 

m2f = realtime_process(
    process=m2f_process,
    outlet=m2f_out,
    deadline=8,
    sample_rate=44100
)
#endregion


#reference signals
upchirp = generate_css_bok_signal([1], symbol_duration, 44100)
downchirp = generate_css_bok_signal([0], symbol_duration, 44100)

_00_qok = generate_css_qok_signal([0,0],  symbol_duration, 44100)
_01_qok = generate_css_qok_signal([0,1],  symbol_duration, 44100)
_10_qok = generate_css_qok_signal([1,0],  symbol_duration, 44100)
_11_qok = generate_css_qok_signal([1,1],  symbol_duration, 44100)

#region Correl filter realtime process
def crl_process(samples):
    # For qok
    # _00_correls = correlation_filter(samples, 44100, _00_qok)[1]
    # _01_correls = correlation_filter(samples, 44100, _01_qok)[1]
    # _10_correls = correlation_filter(samples, 44100, _10_qok)[1]
    # _11_correls = correlation_filter(samples, 44100, _11_qok)[1]
    # return _00_correls, _01_correls, _10_correls, _11_correls

    # For cts
    upchirp_correl = correlation_filter(samples,  44100, upchirp)[1]
    downchirp_correl = correlation_filter(samples,  44100, downchirp)[1]
    return upchirp_correl,  downchirp_correl



def crl_output(samples):
    global fft_fig
    # For Qok
    # _00_correls, _01_correls, _10_correls, _11_correls = samples
    # For cts
    upchirp_correl, downchirp_correl = samples

    if True:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        # Create a new figure for plotting
        fft_fig = plt.figure()
        time_axis = np.arange(len(upchirp_correl)) / 44100  # Create a time axis for plotting
        plt.plot(time_axis, upchirp_correl, label='Filtered Signal', color='blue')
        plt.plot(time_axis, downchirp_correl, label='Filtered Signal', color='red', alpha=0.4)
        plt.grid(True)

        # # Create subplots for each correlation signal
        # fft_fig = plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
        # plt.subplot(2, 2, 1)  # First subplot in a 2x2 grid
        # plt.plot(time_axis, _00_correls, label='Filtered Signal', color='blue')
        # plt.title('Correlation of _00_correls')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.legend()

        # plt.subplot(2, 2, 2)  # Second subplot in a 2x2 grid
        # plt.plot(time_axis, _01_correls, label='Filtered Signal', color='red')
        # plt.title('Correlation of _01_correls')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.legend()

        # plt.subplot(2, 2, 3)  # Third subplot in a 2x2 grid
        # plt.plot(time_axis, _10_correls, label='Filtered Signal', color='orange')
        # plt.title('Correlation of _10_correls')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.legend()

        # plt.subplot(2, 2, 4)  # Fourth subplot in a 2x2 grid
        # plt.plot(time_axis, _11_correls, label='Filtered Signal', color='black')
        # plt.title('Correlation of _11_correls')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.legend()

        # Adjust layout to avoid overlap and show the plot
        plt.tight_layout()
        plt.show()

crl = realtime_process(
    process=crl_process,
    outlet=crl_output,
    deadline=8,
    sample_rate=44100
)
#endregion


#region Narrow-band Filter realtime process
def nbf_process(samples):
    extraband = 0
    result = bandpass_filter(samples, rate, BOK_RANGE[0]+extraband, BOK_RANGE[1]+extraband, 0)
    return result, samples
def nbf_outlet(args):
    global fft_fig
    filtered_samples, original_samples = args

    # Send the filtered signal to the next process
    crl.update(filtered_samples)

    # Plotting logic based on the `plot_nbf` flag
    if False:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        # Create a new figure for plotting
        fft_fig = plt.figure()

        # Create time axis for plotting
        time_axis = np.arange(len(original_samples)) / rate

        # Plot both the original and filtered signals for comparison
        plt.plot(time_axis, original_samples, label='Original Signal', color='orange')
        plt.plot(time_axis, filtered_samples, label='Filtered Signal', color='blue', alpha=0.5)
        plt.title('Original vs Filtered Signal (Narrowband Filter)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()


nbf = realtime_process(
    process=nbf_process,
    outlet=nbf_outlet,
    deadline=8,
    sample_rate=rate
)
#endregion

def update_callback(data):
    data = np.reshape(data,  (-1,))
    nbf.update(data)


listener2.decoder_callbacks.append(update_callback)
listener2.start_listening()
