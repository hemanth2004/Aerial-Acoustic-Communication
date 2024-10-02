import queue
import sys

from scipy.signal import butter, filtfilt, correlate
from signals.resources.gibbs_removal import _gibbs_removal_1d
from signals.util.utils import plot_frequency_spectrum

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import noisereduce as nr

# Configuration Variables
channels = [1]  # Input channels to plot (default: the first)
device = None   # Input device (numeric ID or substring)
window_duration = 10000  # Visible time slot in ms
interval = 30   # Minimum time between plot updates in ms
blocksize = None  # Block size (in samples)
samplerate = None  # Sampling rate of audio device
downsample = 1   # Display every Nth sample

mapping = [c - 1 for c in channels]  # Channel numbers start with 1
q = queue.Queue()

BUFFER_SIZE = int(44100 * 0.5)
buffer = np.zeros((BUFFER_SIZE, len(channels)))
buffer_index = 0  # Index to keep track of position in the buffer

update_flag = False
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)

    data = indata[::downsample, mapping].copy()  # Copy for thread safety
    q.put(data)

    global buffer, buffer_index, update_flag

    num_samples = len(data)

    if buffer_index + num_samples > BUFFER_SIZE:
        end = BUFFER_SIZE - buffer_index
        buffer[buffer_index:] = data[:end]
        buffer[:num_samples - end] = data[end:]
        buffer_index = (buffer_index + num_samples) % BUFFER_SIZE
        update_flag = True
    else:
        buffer[buffer_index:buffer_index + num_samples] = data
        buffer_index = (buffer_index + num_samples) % BUFFER_SIZE


decoder_callbacks = []
update_markers = []

def update_plot(frame):
    """This is called by matplotlib for each plot update."""
    global plotdata
    global lines
    global update_markers
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break

        global buffer, buffer_index, BUFFER_SIZE, update_flag
        if update_flag:
            data = np.array(buffer)

            shift = len(data)
            # Get filtered data from buffer
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data
            
            # Record the update position for marking
            update_markers.append(len(plotdata) - shift)
            for marker in update_markers:
                plt.plot(marker / 44100, plotdata[marker, 0], 'ro')

            if len(decoder_callbacks) != 0:
                for fun in decoder_callbacks:
                    fun(data)

            update_flag = False

    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    
    return lines

def start_listening():
    global plotdata
    global lines
    global BUFFER_SIZE
    global samplerate  # Declare samplerate as global to modify it
    try:
        if samplerate is None:
            device_info = sd.query_devices(device, 'input')
            samplerate = device_info['default_samplerate']
            print("Default rate is ", device_info['default_samplerate'])

        length = int(window_duration * samplerate / (1000 * downsample))
        plotdata = np.zeros((length, len(channels)))

        fig, ax = plt.subplots()
        lines = ax.plot(plotdata)
        if len(channels) > 1:
            ax.legend([f'channel {c}' for c in channels],
                    loc='lower left', ncol=len(channels))
        ax.axis((0, len(plotdata), -1, 1))
        ax.set_yticks([0])
        ax.yaxis.grid(True)
        ax.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
        fig.tight_layout(pad=0)

        stream = sd.InputStream(
            device=device, channels=max(channels),
            samplerate=samplerate, callback=audio_callback)
        ani = FuncAnimation(fig, update_plot, interval=interval, blit=True, cache_frame_data=False)
        with stream:
            plt.show()
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
