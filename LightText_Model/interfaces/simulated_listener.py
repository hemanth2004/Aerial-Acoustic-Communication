import argparse
import queue
import sys
from scipy.signal import butter, filtfilt, correlate
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr

from signals.modulation.ask import generate_ask_signal

SIMULATED_SIGNAL = None
SIMULATED_SAMPLERATE = None

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-w', '--window', type=float, default=10000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args()
q = queue.Queue()

BUFFER_SIZE = 44100 * 0.5
buffer = np.zeros((int(BUFFER_SIZE), 1))
buffer_index = 0
update_flag = False

def audio_callback_simulated(signal, block_size, sample_rate):
    global buffer, buffer_index, update_flag

    num_samples = block_size

    if buffer_index + num_samples > BUFFER_SIZE:
        end = int(BUFFER_SIZE - buffer_index)
        buffer[buffer_index:] = signal[:end].reshape(-1, 1)
        buffer[:num_samples - end] = signal[end:end + (num_samples - end)].reshape(-1, 1)
        buffer_index = (buffer_index + num_samples) % BUFFER_SIZE
        update_flag = True
    else:
        buffer[buffer_index:buffer_index + num_samples] = signal[:num_samples].reshape(-1, 1)
        buffer_index = (buffer_index + num_samples) % BUFFER_SIZE

decoder_callbacks = []
bandpass_flag = True
lowpass_flag = True
update_markers = []

def update_plot(frame):
    global plotdata, lines, update_markers, buffer, buffer_index, BUFFER_SIZE, update_flag

    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break

        if update_flag:
            data = np.array(buffer)
            shift = len(data)
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data

            update_markers.append(len(plotdata) - shift)
            for marker in update_markers:
                plt.plot(marker / SIMULATED_SAMPLERATE, plotdata[marker, 0], 'ro')

            if len(decoder_callbacks) != 0:
                for fun in decoder_callbacks:
                    fun(data)
            update_flag = False
        else:
            continue

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])

    return lines

def start_listening(signal_array=None, samplerate=44100):
    global plotdata, lines, BUFFER_SIZE, SIMULATED_SIGNAL, SIMULATED_SAMPLERATE

    if signal_array is None:
        raise ValueError("Signal array cannot be None for simulation")

    SIMULATED_SIGNAL = signal_array
    SIMULATED_SAMPLERATE = samplerate

    length = int(args.window * samplerate / (1000 * args.downsample))
    plotdata = np.zeros((length, 1))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True, cache_frame_data=False)

    try:
        block_size = 1024
        signal_index = 0

        while signal_index < len(SIMULATED_SIGNAL):
            block = SIMULATED_SIGNAL[signal_index:signal_index + block_size]
            if len(block) < block_size:
                break  # End if we don't have enough samples

            audio_callback_simulated(block.astype(np.float32), block_size, SIMULATED_SAMPLERATE)
            signal_index += block_size

        # Ensure the plot gets updated after all blocks have been processed
        update_flag = True

        # Optionally, call the update_plot function to refresh immediately
        update_plot(0)

    except Exception as e:
        print(f"Error: {e}")

example_signal = generate_ask_signal(
    bit_array=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    sample_rate=44100, symbol_duration=0.2)
def updated_callback(samples):
    print(np.mean(samples))
decoder_callbacks.append(updated_callback)
start_listening(example_signal, 44100)