import interfaces.listener as listener
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from collections import deque

from signals.util.rt import realtime_process
from signals.processing.filters import moving_average, narrowband_filter, bandpass_filter
from interfaces.signals.processing.synchronization_ask import sync_ask
from signals.modulation.ask import decode_ask_signal, generate_ask_signal 
from frame_processing import get_frame

# engine on parameters
power_mean_threshold = 0.001 # for pre preamble phase
power_max_threshold = 0.01 # for pre and post preamble phase

rate = 44100
symbol_duration = 0.2
frame_size = 1 + 5 * 2

in_frame = False

fft_fig = None

#region FFT realtime process
# def ftp_out(output):
#     global fft_fig
#     plt.close(fft_fig)

#     # Calculate magnitude of the FFT output
#     magnitude = np.abs(output)
    
#     # Compute the frequency bins for the FFT output
#     freq = np.fft.fftfreq(len(magnitude), d=1/ftp.sample_rate)

#     # Only take the positive frequencies
#     positive_freq_idx = np.where(freq >= 2)
#     freq = freq[positive_freq_idx]
#     magnitude = magnitude[positive_freq_idx]

#     # Now filter out frequencies above 50 Hz
#     valid_idx = np.where(freq <= 20)  # Get indices where frequency <= 20 Hz
#     filtered_freq = freq[valid_idx]
#     filtered_magnitude = magnitude[valid_idx]

#     fft_fig = plt.figure()
#     plt.plot(filtered_freq, filtered_magnitude)
#     plt.title('FFT Output (<= 20 Hz)')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.grid(True)
#     plt.show()
# ftp = realtime_process(
#     process=np.fft.fft, 
#     outlet=ftp_out, 
#     deadline=8, 
#     sample_rate=44100
# )
#endregion

#region Differentiate Envelope
def dif_out(samples):
    global fft_fig
    if False:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

            #Plot the envelope
            time_axis = np.arange(len(samples)) / 44100  # Create a time axis for plotting
            fft_fig = plt.figure()
            plt.plot(time_axis, samples, label='d/dx Envelope', color='orange')
            plt.title('d/dx Hilbert Envelope')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.legend()
            plt.show(block=False)  # Non-blocking show to allow for real-time updates
            plt.pause(0.001)  # Small pause to allow the plot to update
    return
dif = realtime_process(
    process=np.gradient,
    outlet=dif_out, 
    deadline=8,
    sample_rate=44100
)
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

syncing = sync_ask(
    demod=decode_ask_signal,
    sampling_rate=44100,
    symbol_duration=symbol_duration,
    frame_outlet=frame_outlet,
    preamble=np.array([1, 1, 1, 1]),
    plot=False,
    bottleneck_deadline=8,
    fixed_frame_size=frame_size
)

hbe_engine_decision = False
#endregion

#region Interruption Filter3 realtime process
recent_inf2_samples = np.array([])
recent_inf1_samples = np.array([])
def inf3_process(samples):
    symbol_duration_samples = int(syncing.symbol_duration * syncing.sampling_rate)
    threshold_length = int(0.50 * symbol_duration_samples)

    # Step 1: Determine the indices of the initial and trailing contiguous blocks
    first_transition = 0
    last_transition = len(samples) - 1

    # Find the first transition point where a change in value occurs
    while first_transition < len(samples) - 1 and samples[first_transition] == samples[first_transition + 1]:
        first_transition += 1

    # Find the last transition point where a change in value occurs
    while last_transition > 0 and samples[last_transition] == samples[last_transition - 1]:
        last_transition -= 1

    # Ensure to keep the initial and trailing blocks intact in the final result
    filtered_samples = samples.copy()

    # Only process the samples between the first and last transition
    i = first_transition + 1

    # Step 2: Traverse through the samples to find blocks of 0s and 1s in the core region
    while i < last_transition:
        current_bit = samples[i]
        start = i

        # Find the end of the current block of identical bits
        while i < len(samples) and samples[i] == current_bit:
            i += 1

        # Determine the length of the current block
        block_length = i - start

        # If the block is surrounded by the opposite bits and the block length is below the threshold, neutralize it
        if start > first_transition and i < last_transition:
            if samples[start - 1] != current_bit and samples[i] != current_bit:
                if block_length < threshold_length:
                    check_length = min(int(threshold_length / 0.6), start - first_transition, last_transition - i)

                    left_mean = np.mean(samples[start - check_length:start])
                    right_mean = np.mean(samples[i:i + check_length])

                    target_bit = 0

                    if left_mean > 0.5 and right_mean > 0.5:
                        target_bit = 1
                    elif left_mean < 0.5 and right_mean < 0.5:
                        target_bit = 0
                    elif left_mean > 0.5 and right_mean < 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 1 if left_mean > right_mean else 0
                    elif left_mean < 0.5 and right_mean > 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 0 if left_mean > right_mean else 1

                    # Neutralize by setting to surrounding bit
                    filtered_samples[start:i] = target_bit

        # If the block is at the beginning or end, check against the threshold as well
        elif (start == first_transition and block_length < threshold_length and i < len(samples)) or \
             (i == last_transition and block_length < threshold_length and start > 0):
            filtered_samples[start:i] = samples[start - 1] if start > 0 else samples[i - 1]

    return filtered_samples, samples

def inf3_out(args):
    global fft_fig, recent_inf1_samples, recent_inf2_samples
    samples, samples_original = args

    syncing.append_samples(samples)

    if True:
        # Close the previous figure if it exists
        if fft_fig is not None:
            plt.close(fft_fig)

        # Create a new figure
        fft_fig = plt.figure()
        time_axis = np.arange(len(samples_original)) / syncing.sampling_rate # Create a time axis for plotting

        # Plot the original and filtered signals
        plt.plot(time_axis, recent_inf1_samples*0.7, label='INF 1 Input', color='brown')
        plt.plot(time_axis, recent_inf2_samples*0.8, label='INF 2 Input', color='red')
        plt.plot(time_axis, samples_original*0.9, label='INF 3 Input', color='orange')
        plt.plot(time_axis, samples, label='Filtered Signal', color='blue')  # Adjust alpha for visibility
        ax = plt.gca()
        ax.set_ylim([-1.2, 1.2])
        plt.title('Interruption Filter [Sync Engine: {status}]'.format(status = syncing.engine_on))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        fft_fig.canvas.manager.window.wm_geometry("+1100+200")

inf3 = realtime_process(
    process=inf3_process,
    outlet=inf3_out,
    deadline=8,
    sample_rate=44100
)
#endregion

#region Interruption Filter2 realtime process
recent_inf1_samples = np.array([])
def inf2_process(samples):
    symbol_duration_samples = int(syncing.symbol_duration * syncing.sampling_rate)
    threshold_length = int(0.5 * symbol_duration_samples)

    # Step 1: Determine the indices of the initial and trailing contiguous blocks
    first_transition = 0
    last_transition = len(samples) - 1

    # Find the first transition point where a change in value occurs
    while first_transition < len(samples) - 1 and samples[first_transition] == samples[first_transition + 1]:
        first_transition += 1

    # Find the last transition point where a change in value occurs
    while last_transition > 0 and samples[last_transition] == samples[last_transition - 1]:
        last_transition -= 1

    # Ensure to keep the initial and trailing blocks intact in the final result
    filtered_samples = samples.copy()

    # Only process the samples between the first and last transition
    i = first_transition + 1

    # Step 2: Traverse through the samples to find blocks of 0s and 1s in the core region
    while i < last_transition:
        current_bit = samples[i]
        start = i

        # Find the end of the current block of identical bits
        while i < len(samples) and samples[i] == current_bit:
            i += 1

        # Determine the length of the current block
        block_length = i - start

        # If the block is surrounded by the opposite bits and the block length is below the threshold, neutralize it
        if start > first_transition and i < last_transition:
            if samples[start - 1] != current_bit and samples[i] != current_bit:
                if block_length < threshold_length:
                    check_length = min(int(threshold_length / 0.6), start - first_transition, last_transition - i)

                    left_mean = np.mean(samples[start - check_length:start])
                    right_mean = np.mean(samples[i:i + check_length])

                    target_bit = 0

                    if left_mean > 0.5 and right_mean > 0.5:
                        target_bit = 1
                    elif left_mean < 0.5 and right_mean < 0.5:
                        target_bit = 0
                    elif left_mean > 0.5 and right_mean < 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 1 if left_mean > right_mean else 0
                    elif left_mean < 0.5 and right_mean > 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 0 if left_mean > right_mean else 1

                    # Neutralize by setting to surrounding bit
                    filtered_samples[start:i] = target_bit

        # If the block is at the beginning or end, check against the threshold as well
        elif (start == first_transition and block_length < threshold_length and i < len(samples)) or \
             (i == last_transition and block_length < threshold_length and start > 0):
            filtered_samples[start:i] = samples[start - 1] if start > 0 else samples[i - 1]

    return filtered_samples, samples

def inf2_out(args):
    global fft_fig, recent_inf1_samples, recent_inf2_samples
    samples, samples_original = args

    recent_inf2_samples = samples_original
    inf3.update(samples)

    if False:
        # Close the previous figure if it exists
        if fft_fig is not None:
            plt.close(fft_fig)

        # Create a new figure
        fft_fig = plt.figure()
        time_axis = np.arange(len(samples_original)) / syncing.sampling_rate # Create a time axis for plotting

        # Plot the original and filtered signals
        plt.plot(time_axis, recent_inf1_samples*0.8, label='INF 1 Input', color='brown')
        plt.plot(time_axis, samples_original*0.9, label='INF 2 Input', color='orange')
        plt.plot(time_axis, samples*1.1, label='Filtered Signal', color='blue', alpha=0.5)  # Adjust alpha for visibility
        plt.title('Interruption Filter [Sync Engine: {status}]'.format(status = hbe_engine_decision))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        fft_fig.canvas.manager.window.wm_geometry("+1100+10")

inf2 = realtime_process(
    process=inf2_process,
    outlet=inf2_out,
    deadline=8,
    sample_rate=44100
)
#endregion

#region Interruption Filter1 realtime process
def inf_process(samples):
    symbol_duration_samples = int(syncing.symbol_duration * syncing.sampling_rate)
    threshold_length = int(0.45 * symbol_duration_samples)

    # Step 1: Determine the indices of the initial and trailing contiguous blocks
    first_transition = 0
    last_transition = len(samples) - 1

    # Find the first transition point where a change in value occurs
    while first_transition < len(samples) - 1 and samples[first_transition] == samples[first_transition + 1]:
        first_transition += 1

    # Find the last transition point where a change in value occurs
    while last_transition > 0 and samples[last_transition] == samples[last_transition - 1]:
        last_transition -= 1

    # Ensure to keep the initial and trailing blocks intact in the final result
    filtered_samples = samples.copy()

    # Only process the samples between the first and last transition
    i = first_transition + 1

    # Step 2: Traverse through the samples to find blocks of 0s and 1s in the core region
    while i < last_transition:
        current_bit = samples[i]
        start = i

        # Find the end of the current block of identical bits
        while i < len(samples) and samples[i] == current_bit:
            i += 1

        # Determine the length of the current block
        block_length = i - start

        # If the block is surrounded by the opposite bits and the block length is below the threshold, neutralize it
        if start > first_transition and i < last_transition:
            if samples[start - 1] != current_bit and samples[i] != current_bit:
                if block_length < threshold_length:
                    check_length = min(int(threshold_length / 0.6), start - first_transition, last_transition - i)

                    left_mean = np.mean(samples[start - check_length:start])
                    right_mean = np.mean(samples[i:i + check_length])

                    target_bit = 0

                    if left_mean > 0.5 and right_mean > 0.5:
                        target_bit = 1
                    elif left_mean < 0.5 and right_mean < 0.5:
                        target_bit = 0
                    elif left_mean > 0.5 and right_mean < 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 1 if left_mean > right_mean else 0
                    elif left_mean < 0.5 and right_mean > 0.5:
                        left_mean = np.abs(left_mean - 0.5)
                        right_mean = np.abs(right_mean - 0.5)
                        target_bit = 0 if left_mean > right_mean else 1

                    # Neutralize by setting to surrounding bit
                    filtered_samples[start:i] = target_bit

        # If the block is at the beginning or end, check against the threshold as well
        elif (start == first_transition and block_length < threshold_length and i < len(samples)) or \
             (i == last_transition and block_length < threshold_length and start > 0):
            filtered_samples[start:i] = samples[start - 1] if start > 0 else samples[i - 1]

    return filtered_samples, samples

def inf_out(args):
    global fft_fig, recent_inf1_samples
    samples, samples_original = args

    recent_inf1_samples = samples_original
    inf2.update(samples)

    if False:
        # Close the previous figure if it exists
        if fft_fig is not None:
            plt.close(fft_fig)

        # Create a new figure
        fft_fig = plt.figure()
        time_axis = np.arange(len(samples_original)) / syncing.sampling_rate # Create a time axis for plotting

        # Plot the original and filtered signals
        plt.plot(time_axis, samples_original*0.9, label='Original Signal', color='orange')
        plt.plot(time_axis, samples*1.1, label='Filtered Signal', color='blue', alpha=0.5)  # Adjust alpha for visibility
        plt.title('INF 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()

inf = realtime_process(
    process=inf_process,
    outlet=inf_out,
    deadline=8,
    sample_rate=44100
)
#endregion

#region Comparator1 realtime process

def cmp1_process(samples):
    threshold = np.mean(samples) * 0.9
    # Compare when sample value > threshold
    return (samples > threshold).astype(int), samples

def cmp1_out(params):
    samples, original = params
    global hbe_engine_decision

    # # Count the variance in block sizes to know whether its bits or random signals
    # def count_contiguous_blocks(signal):
    #     counts_0 = []
    #     counts_1 = []
    #     current_count = 1
    #     for i in range(1, len(signal)):
    #         if signal[i] == signal[i - 1]:
    #             current_count += 1  # Increment count if the same as previous
    #         else:
    #             # Append the count to the respective list
    #             if signal[i - 1] == 0:
    #                 counts_0.append(current_count)
    #             else:
    #                 counts_1.append(current_count)
    #             current_count = 1  # Reset count for the new value

    #     # Append the last counted block
    #     if signal[-1] == 0:
    #         counts_0.append(current_count)
    #     else:
    #         counts_1.append(current_count)

    #     return np.array(counts_0), np.array(counts_1)
    # count0s, count1s = count_contiguous_blocks(samples)
    # total = np.concatenate((count0s, count1s))
    # variance = np.var(total)
    # def format_indian_number(n):
    #     return f"{n:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

    # # print("Variance: ", format_indian_number(variance), "\nhilbert decision: ", hbe_engine_decision)
    # # print()

    syncing.set_engine_status(hbe_engine_decision)
    inf.update(samples)

    global fft_fig
    if False:
        max_value = np.max(np.abs(original))
        normalized_original = original / max_value if max_value != 0 else original  # Avoid division by zero
        
        
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        dif.update(samples)

        fft_fig = plt.figure()
        time_axis = np.arange(len(original)) / 44100  # Create a time axis for plotting
        plt.plot(time_axis, normalized_original, label='Envelope', color='orange')
        plt.plot(time_axis, samples*0.5, label='Binary Output', color='blue', alpha=0.5)  # Adjust alpha for better visibility
        plt.title('Comparator Output')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude / Binary Output (0 or 1)')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return

cmp1 = realtime_process(
    process=cmp1_process,
    outlet=cmp1_out,
    deadline=8,
    sample_rate=44100
)
#endregion

#region Hilbert Envelope realtime process
def hbe_process(samples):
    analytical_signal = hilbert(samples)
    envelope = np.abs(analytical_signal)
    envelope = moving_average(envelope, window_size=1000)
    return envelope, samples
def hbe_out(args):
    envelope, original_samples = args
    global hbe_engine_decision

    signal_mean = np.mean(envelope)
    signal_max = np.max(envelope)
    print("-------\nMean = ", signal_mean, "\nMax = ", signal_max)
    # if syncing.post_preamble:
    #     hbe_engine_decision = signal_max > power_max_threshold
    # else:
    #     hbe_engine_decision = signal_max > power_max_threshold #and signal_mean > power_mean_threshold

    cmp1.update(envelope)

    hbe_engine_decision = signal_max > power_max_threshold
    global fft_fig
    if False:
        if fft_fig is not None:
            plt.close(fft_fig)  # Close the previous figure if it exists

        time_axis = np.arange(len(envelope)) / 44100  # Create a time axis for plotting
        orig_time_axis = np.arange(len(original_samples)) / 44100
        fft_fig = plt.figure()
        plt.plot(orig_time_axis, original_samples, label='Original', color='pink')
        plt.plot(time_axis, envelope, label='Envelope', color='orange')
        ax = plt.gca()
        ax.set_ylim([-0.05, 0.05])
        plt.title('Hilbert Envelope')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show(block=False)  # Non-blocking show to allow for real-time updates
        fft_fig.canvas.manager.window.wm_geometry("+1100+200")

hbe = realtime_process(
    process=hbe_process,
    outlet=hbe_out,
    deadline=8,
    sample_rate=44100
)
#endregion

#region Narrow-band Filter realtime process
def nbf_process(samples):
    result = bandpass_filter(samples, rate, 3600, 3800, 3)
    return result, samples
def nbf_outlet(args):
    global fft_fig
    filtered_samples, original_samples = args

    # Send the filtered signal to the next process
    hbe.update(filtered_samples)

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


listener.decoder_callbacks.append(update_callback)
listener.start_listening()
