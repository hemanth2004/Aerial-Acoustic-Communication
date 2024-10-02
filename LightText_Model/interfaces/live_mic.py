"""
Handles Real-Time Signal Reception and Demodulation
"""

# signal_reception.py
import time
import numpy as np
import pandas as pd
import pyaudio

p = None
stream = None

def on_awake():
    global p, stream

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)
    
def update(delta_time):
    # global stream

    # # Read data from the microphone
    # data = stream.read(1024, exception_on_overflow=False)
    # mic_values = np.frombuffer(data, dtype=np.int16)
    
    # print(f"Input latency: {stream.get_input_latency()}")
    # print(f"CPU load: {stream.get_cpu_load()}")
    # print(f"Mic Values: {pd.Series(mic_values)}") 

    # fps = 1.0 / delta_time
    # print(f"FPS: {fps:.2f}")
    pass

def on_exit():
    global p, stream

    # Close the stream and PyAudio instance when done
    if stream is not None:
        stream.stop_stream()
        stream.close()
    if p is not None:
        p.terminate()

    print("Exiting program...")

def main_loop(callback, max_delta_time=0.1):
    previous_time = time.time()
    running = True

    on_awake()

    while running:
        current_time = time.time()
        delta_time = current_time - previous_time
        previous_time = current_time

        if delta_time > max_delta_time:
            delta_time = max_delta_time

        if delta_time == 0:
            delta_time = 1e-6

        global stream
        callback(delta_time, stream)

        time.sleep(0.000001)

    on_exit()
