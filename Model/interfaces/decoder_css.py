import listener
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from collections import deque

from signals.processing.synchronization_css import sync_css
from signals.util.rt import realtime_process
from signals.util.utils import scale_signal
from signals.processing.filters import moving_average, sos_butter_bandpass_filter, bandpass_filter, correlation_filter, matched_filter
from signals.modulation.css import generate_css_bok_signal, generate_css_qok_signal, generate_css_bok_signal_half_range, generate_css_cts_signal
from signals.modulation.css import CTS_RANGE, BOK_RANGE, QOK_RANGE, QOK_SWEEP1
from frame_processing import get_frame

import psutil
import time
import threading

def monitor_system_usage(interval=1.0):
    """
    Monitors and prints CPU and memory usage of the current program at regular intervals.

    Parameters:
    - interval (float): Time in seconds between each usage check.
    """
    # Get the current process
    process = psutil.Process(os.getpid())
    
    try:
        while True:
            # Get CPU and memory usage
            cpu_usage = process.cpu_percent(interval=interval)
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / (1024 ** 2)  # Convert bytes to MB

            # Display the CPU and memory usage
            print(f"CPU Usage: {cpu_usage}%")
            print(f"Memory Usage: {memory_usage:.2f} MB")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

# Start the monitor function in a new thread
monitor_thread = threading.Thread(target=monitor_system_usage, args=(1.0,), daemon=True)
# monitor_thread.start()

# engine on parameters
power_mean_threshold = 0.3
power_max_threshold = 1 # for pre and post preamble phase

rate = 44100
symbol_duration = 0.13
symbol_gap = 0.03
frame_size = 5*2

main_fig = None

#region Utils
def preprocess_signal(signal):
    # Create a copy of the signal to avoid modifying the original
    processed_signal = np.zeros_like(signal)
    
    # Initialize the start of a contiguous block
    start = None
    
    for i in range(len(signal)):
        if signal[i] > 0:
            # Start of a contiguous block of 1s
            if start is None:
                start = i
        else:
            # End of a contiguous block of 1s
            if start is not None:
                # Calculate the midpoint of the block
                midpoint = (start + i - 1) // 2
                processed_signal[midpoint] = 1
                start = None
    
    # Handle case where the last block extends to the end of the signal
    if start is not None:
        midpoint = (start + len(signal) - 1) // 2
        processed_signal[midpoint] = 1
    
    return processed_signal

def combine_signals(signal1, signal2):
    # Ensure both signals have the same number of samples
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same number of samples.")
    
    # Preprocess both signals to retain only the midpoint of contiguous 1s blocks
    processed_signal1 = preprocess_signal(signal1)
    processed_signal2 = preprocess_signal(signal2)
    
    # Start with the processed first signal and overwrite with -1 where the second has 1s
    combined_signal = np.copy(processed_signal1)
    combined_signal[processed_signal2 > 0] = -1
    
    return combined_signal

#endregion

#region Synchronization Engine
def frame_outlet(frame):
    f = open("__comm.txt", "r")
    orig = f.read()
    orig_bits = np.array([])
    for i in orig:
        orig_bits = np.append(orig_bits, [int(i)])

    frame = frame.astype(int)
    orig_bits = orig_bits.astype(int)
    get_frame(frame, orig_bits)

def bit_outlet(bit):
    # print("bit detected ", bit)
    pass
syncing = sync_css(
    sampling_rate=rate,
    symbol_duration=symbol_duration + symbol_gap,
    bit_outlet=None,
    frame_outlet=frame_outlet,
    preamble=[1, 1, 0, 1],
    plot=False,
    bottleneck_deadline=8,
    frame_size=frame_size
)
#endregion

#region Signal Combination realtime process
def scm_process(samples):
    uc_samples, dc_samples = samples
    return combine_signals(uc_samples, dc_samples)

def scm_out(samples):
    global main_fig

    syncing.append_samples(samples)
    # Plotting logic based on the `plot_nbf` flag
    if True:
        if main_fig is not None:
            plt.close(main_fig)  # Close the previous figure if it exists

        # Create a new figure for plotting
        main_fig = plt.figure()

        # Create time axis for plotting
        time_axis = np.arange(len(samples)) / rate

        # Plot both the original and filtered signals for comparison
        plt.plot(time_axis, samples, color='orange')
        plt.title('Combined Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()


scm = realtime_process(
    process=scm_process,
    outlet=scm_out,
    deadline=8,
    sample_rate=rate,
    signals=2
)
#endregion

#region Differentiate Envelope
def dif_process(samples):
    uc_samples, dc_samples = samples
    uc_grad = np.gradient(uc_samples)
    dc_grad = np.gradient(dc_samples)

    return np.maximum(uc_grad, 0), np.maximum(dc_grad, 0)

def dif_out(samples):
    uc_samples, dc_samples = samples
    #print("upchirp lines:\n", "even" if len(uc_samples[uc_samples > 0])%2==0 else "odd")
    #print("downchirp lines:\n", "even" if len(dc_samples[dc_samples > 0])%2==0 else "odd")
    scm.update(samples)

    global main_fig
    if False:
        
        
        if main_fig is not None:
            plt.close(main_fig)  # Close the previous figure if it exists

        main_fig = plt.figure()

        #Plot the envelope
        time_axis = np.arange(len(uc_samples))  # Create a time axis for plotting
        plt.plot(time_axis, dc_samples, label='downchirps', color='red')
        plt.plot(time_axis, uc_samples, label='upchirp', color='blue')
        plt.title('d/dx peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Peaks')
        plt.grid(True)
        plt.legend()
        plt.show()

dif = realtime_process(
    process=dif_process,
    outlet=dif_out, 
    deadline=8,
    sample_rate=rate,
    signals=2
)
#endregion

#region Comparator realtime process
def cmp_process(samples):
    uc_samples, dc_samples = samples
    uc_threshold_mid = (np.max(uc_samples) - np.mean(uc_samples)) * 0.6
    dc_threshold_mid = (np.max(dc_samples) - np.mean(dc_samples)) * 0.6

    return (uc_samples > uc_threshold_mid).astype(int), (dc_samples > dc_threshold_mid).astype(int)

def cmp_out(samples):
    
    dif.update(samples)
    global main_fig
    if False:
        uc_samples, dc_samples = samples
        if main_fig is not None:
            plt.close(main_fig)  # Close the previous figure if it exists

        main_fig = plt.figure()
        time_axis = np.arange(len(uc_samples)) / rate  # Create a time axis for plotting
        plt.plot(time_axis, uc_samples, label='upchirp', color='blue')
        plt.plot(time_axis, dc_samples, label='downchirps', color='red')
        plt.title('Comparator Output')
        plt.xlabel('Time (s)')
        plt.ylabel('Chirp Blocks')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)  # Non-blocking show to allow for real-time updates
        main_fig.canvas.manager.window.wm_geometry("+1100+200")

cmp = realtime_process(
    process=cmp_process,
    outlet=cmp_out,
    deadline=8,
    sample_rate=rate,
    signals=2
)
#endregion

#region Hilbert Envelope realtime process
def hbe_process(samples):
    uc_samples, dc_samples = samples

    uc_analytical_signal = hilbert(uc_samples)
    uc_envelope = np.abs(uc_analytical_signal)
    uc_envelope = moving_average(uc_envelope, window_size=100)

    dc_analytical_signal = hilbert(dc_samples)
    dc_envelope = np.abs(dc_analytical_signal)
    dc_envelope = moving_average(dc_envelope, window_size=100)
    return scale_signal(uc_envelope), scale_signal(dc_envelope)

def hbe_out(samples):
    uc_samples, dc_samples = samples
    cmp.update(samples)

    uc_threshold_mid = (np.max(uc_samples) - np.mean(uc_samples)) * 0.6
    dc_threshold_mid = (np.max(dc_samples) - np.mean(dc_samples)) * 0.6

    # mean_filtered = np.mean(uc_samples + dc_samples)
    # max_filtered = np.max(uc_samples + dc_samples)
    # print("max ", max_filtered)
    # print("mean ", mean_filtered)
    # print("engine status: ", syncing.engine_on)
    # status = max_filtered > power_max_threshold and mean_filtered > power_mean_threshold
    # syncing.set_engine_status(status)

    global main_fig
    if False:
        if main_fig is not None:
            plt.close(main_fig)  # Close the previous figure if it exists

        time_axis = np.arange(len(uc_samples)) / rate  # Create a time axis for plotting
        main_fig = plt.figure()
                
        plt.axhline(y=uc_threshold_mid, color='blue', linestyle='--', label='Upchirp Threshold')
        plt.axhline(y=dc_threshold_mid, color='red', linestyle='--', label='Downchirp Threshold')

        plt.plot(time_axis, dc_samples, label='downchirp', color='red')
        plt.plot(time_axis, uc_samples, label='upchirp', color='blue')
        plt.title('Hilbert Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)  # Non-blocking show to allow for real-time updates
        main_fig.canvas.manager.window.wm_geometry("+1100+200")

hbe = realtime_process(
    process=hbe_process,
    outlet=hbe_out,
    deadline=8,
    sample_rate=rate,
    signals=2
)
#endregion

#region reference signals
upchirp = generate_css_bok_signal([1], symbol_duration, rate)
downchirp = generate_css_bok_signal([0], symbol_duration, rate)

_00_qok = generate_css_qok_signal([0,0],  symbol_duration, rate)
_01_qok = generate_css_qok_signal([0,1],  symbol_duration, rate)
_10_qok = generate_css_qok_signal([1,0],  symbol_duration, rate)
_11_qok = generate_css_qok_signal([1,1],  symbol_duration, rate)
#endregion

#region Correl filter realtime process
def crl_process(samples):
    # For qok
    # _00_correls = correlation_filter(samples, rate, _00_qok)[1]
    # _01_correls = correlation_filter(samples, rate, _01_qok)[1]
    # _10_correls = correlation_filter(samples, rate, _10_qok)[1]
    # _11_correls = correlation_filter(samples, rate, _11_qok)[1]
    # return _00_correls, _01_correls, _10_correls, _11_correls

    # For cts
    upchirp_correl = matched_filter(samples,  rate, upchirp)[1]
    downchirp_correl = matched_filter(samples,  rate, downchirp)[1]
    return upchirp_correl, downchirp_correl

def crl_output(samples):
    global main_fig

    # For Qok
    # _00_correls, _01_correls, _10_correls, _11_correls = samples
    # For cts
    upchirp_correl, downchirp_correl = samples

    hbe.update(samples)

    if False:
        if main_fig is not None:
            plt.close(main_fig)  
        main_fig = plt.figure()
        time_axis = np.arange(len(upchirp_correl)) / rate

        plt.plot(time_axis, upchirp_correl, label='Upchirp Correl', color='blue')
        plt.plot(time_axis, downchirp_correl, label='Downchirp Correl', color='red', alpha=0.8)
        plt.title("Correlation output")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

crl = realtime_process(
    process=crl_process,
    outlet=crl_output,
    deadline=8,
    sample_rate=rate
)
#endregion

#region Narrow-band Filter realtime process
def nbf_process(samples):
    extraband = 0
    result = sos_butter_bandpass_filter(samples, BOK_RANGE[0]-extraband, BOK_RANGE[1]+extraband, rate, 5)
    return result, samples
def nbf_outlet(args):
    global main_fig
    filtered_samples, original_samples = args
    crl.update(filtered_samples)

    # Plotting logic based on the `plot_nbf` flag
    if False:
        if main_fig is not None:
            plt.close(main_fig)  # Close the previous figure if it exists

        # Create a new figure for plotting
        main_fig = plt.figure()

        # Create time axis for plotting
        time_axis = np.arange(len(original_samples)) / rate

        # Plot both the original and filtered signals for comparison
        plt.plot(time_axis, original_samples, label='Original Signal', color='orange')
        plt.plot(time_axis, filtered_samples, label='Filtered Signal', color='blue')
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

listener.decoder_callbacks.append(update_callback)
listener.start_listening()
