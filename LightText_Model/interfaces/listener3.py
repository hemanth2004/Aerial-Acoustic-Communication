import queue
import sys
import sounddevice as sd
import numpy as np
import threading
import matplotlib.pyplot as plt

# Configuration Variables
channels = [1]  # Input channels to plot (default: the first)
device = None   # Input device (numeric ID or substring)
window_duration = 10000  # Visible time slot in ms
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

def process_data():
    """Process the audio data received in the callback."""
    global update_flag
    while True:
        try:
            data = q.get_nowait()
            if update_flag:
                global buffer
                buffer_data = np.array(buffer)

                # Here you can process the buffer_data as needed
                # For example, you could call a decoder or filter function
                for callback in decoder_callbacks:
                    callback(buffer_data)

                update_flag = False
        except queue.Empty:
            continue


def start_listening():
    global samplerate  # Declare samplerate as global to modify it
    try:
        if samplerate is None:
            device_info = sd.query_devices(device, 'input')
            samplerate = device_info['default_samplerate']
            print("Default rate is ", device_info['default_samplerate'])

        stream = sd.InputStream(
            device=device, channels=max(channels),
            samplerate=samplerate, callback=audio_callback)
        
        # Start the processing thread
        threading.Thread(target=process_data, daemon=True).start()
        
        # Start the plotting in the main thread
        with stream:
            print("Listening to audio input...")
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

# To start the listening process, call start_listening()
