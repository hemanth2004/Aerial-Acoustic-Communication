import time
import os
import pyaudio
import numpy as np
import keyboard

# Global variables
p = None
stream = None

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

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
    global stream

    # Read data from the microphone
    data = stream.read(1024, exception_on_overflow=False)
    mic_values = np.frombuffer(data, dtype=np.int16)

    # Update Loop with microphone values
    clear_terminal()
    print(f"Input latency: {stream.get_input_latency()}")
    print(f"CPU load: {stream.get_cpu_load()}")
    print(f"Mic Values: {mic_values[:10]}")  # Display the first 10 mic values as a sample

    # Calculate FPS
    fps = 1.0 / delta_time
    print(f"FPS: {fps:.2f}")

def on_exit():
    global p, stream

    # Close the stream and PyAudio instance when done
    if stream is not None:
        stream.stop_stream()
        stream.close()
    if p is not None:
        p.terminate()

    print("Exiting program...")

def main():
    max_delta_time = 0.00001
    previous_time = time.time()
    running = True

    on_awake()

    # Function to exit the loop
    def exit_program():
        nonlocal running
        running = False

    # Register the shortcut (Left Shift + Alt)
    keyboard.add_hotkey('shift+alt', exit_program)

    while running:
        current_time = time.time()
        delta_time = current_time - previous_time
        previous_time = current_time

        if delta_time > max_delta_time:
            delta_time = max_delta_time

        if delta_time == 0:
            delta_time = 1e-6

        update(delta_time)

        time.sleep(0.000001)

    on_exit()

if __name__ == "__main__":
    main()
