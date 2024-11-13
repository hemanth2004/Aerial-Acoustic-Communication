import numpy as np
import matplotlib.pyplot as plt
import threading
from collections import deque

class sync_css:
    processed_in_this_frame = 0
    post_preamble = False
    engine_on = True

    sample_deque = deque()

    sampling_rate = 44100
    min_samples = sampling_rate * 1
    preamble = np.array([])
    samples_limit = sampling_rate * 32

    def __init__(self, sampling_rate, symbol_duration, preamble, frame_size, bit_outlet=None, frame_outlet=None, plot=True, bottleneck_deadline=2):
        self.sampling_rate = sampling_rate
        self.symbol_duration = symbol_duration

        self.frame_outlet = frame_outlet
        self.bit_outlet = bit_outlet

        self.frame_size = frame_size
        self.preamble = preamble
        self.plot = plot
        self.bottleneck_deadline = bottleneck_deadline
        self.preamble_buffer = FixedLengthQueue(max_size=len(self.preamble))  # Use CircularBuffer for preamble detection
        self.bit_buffer = FixedLengthQueue(max_size=self.frame_size)
        self.signal_cursor = 0

    def set_engine_status(self, status):
        if status:
            self.engine_on = True
        else:
            if not self.engine_on:
                return
            t = self.bottleneck_deadline * 1.5
            timer = threading.Timer(t, self.__turn_off_engine)
            timer.start()

    def __turn_off_engine(self):
        self.engine_on = False

    def append_samples(self, samples):
        if self.engine_on:
            self.sample_deque.extend(samples)

        self.ensure_mem()
        if self.ensure_min():
            self.update()

    def update(self):
        if not self.post_preamble:  # Pre-preamble phase: synchronization and waiting for end of preamble
            self.post_preamble = self.detect_preamble()
            if self.post_preamble:
                self.processed_in_this_frame = 0  # Reset symbol counter
                self.process_post_preamble()
        else:  # Post-preamble phase: process frame symbols
            self.process_post_preamble()

        if self.plot:
            self.plot_signal()

    def detect_preamble(self):
        signal = np.array(self.sample_deque)
        print("cursor target :", self.signal_cursor)
        for i in range(0, len(signal)):
            sample = signal[i]
            if sample != 0 and (sample == 1 or sample == -1):
                """
                we enque both index and sample to make sure that once we match 
                buffer to preamble, we can also verify if each of the preamble bits 
                are spaced symbol duration apart (% error margin on either side)
                also possibly combine any other symbol within half a symbol duration 
                on either side of the detected sample and keep only the symbol (1 or -1) 
                thats the most possible bit
                """

                # cleanup the right surroundings of the detected bit to make sure theres
                # only 1 sample in the locality
                cleanup_radius = int(0.9 * self.symbol_duration * self.sampling_rate)

                list_of_detections = []
                for j in range(i, min(len(signal), i+cleanup_radius)):
                    if signal[j] != 0 and (signal[j] == 1 or signal[j] == -1):
                        list_of_detections.append(signal[j])
                        # remove the detected symbol from the signal
                        self.sample_deque[j] = 0

                
                #if len(list_of_detections) > 1:
                #print("\ncombined symbols ", list_of_detections)
                detected_mean = np.mean(list_of_detections)

                detected_bit = sample
                if detected_mean > 0:
                    detected_bit = 1
                elif detected_mean < 0:
                    detected_bit = -1
                
                # print("\n\nlist of detections: ", list_of_detections, "\nmean: ", detected_mean)

                self.sample_deque[i] = detected_bit

                bit = 1 if detected_bit == 1 else 0
                self.preamble_buffer.enqueue([i, bit])
                
                # also add code to remove any other sample in half a symbol duration radius
                # print("current prepreamble buffer: ", self.preamble_buffer)
            
            if self.preamble_buffer.size() == len(self.preamble):
                # check if the buffer matches the preamble
                circular_buffer_bits = [samp[1] for samp in self.preamble_buffer]

                if circular_buffer_bits == self.preamble:
                    # expected minimum distance between preamble bits as a fraction of symbol_duration
                    distance_threshold = 0.5 * self.symbol_duration * self.sampling_rate
                    distances = [(self.preamble_buffer[j + 1][0] - self.preamble_buffer[j][0]) for j in range(self.preamble_buffer.size() - 1)]

                    if all(dist >= distance_threshold for dist in distances):
                        # preamble detected
                        self.signal_cursor = i+1 # cursor is at exact instant the preamble ends
                        print("\nDETECTED PREAMBLE 1101\n")
                        return True
        return False

    def process_post_preamble(self): 
        signal = np.array(self.sample_deque)
        print("cursor target :", self.signal_cursor)
        for i in range(self.signal_cursor, len(signal)):
            sample = signal[i]
            if sample != 0 and (sample == 1 or sample == -1):
                """
                follow mostly the same strategy as preamble but with a smaller margin
                """
                cleanup_radius = int(0.9 * self.symbol_duration * self.sampling_rate)

                list_of_detections = []
                for j in range(i, min(len(signal), i+cleanup_radius)):
                    if signal[j] != 0 and (signal[j] == 1 or signal[j] == -1):
                        list_of_detections.append(signal[j])
                        # remove the detected symbol from the signal
                        self.sample_deque[j] = 0
                    detected_mean = np.mean(list_of_detections)

                detected_bit = sample
                if detected_mean > 0:
                    detected_bit = 1
                elif detected_mean < 0:
                    detected_bit = -1
                
                # print("\n\nlist of detections: ", list_of_detections, "\nmean: ", detected_mean)

                self.sample_deque[i] = detected_bit

                bit = 1 if detected_bit == 1 else 0
                self.bit_buffer.enqueue([i, bit])
                if self.bit_outlet is not None:
                    self.bit_outlet(bit)
            
            if self.bit_buffer.is_full():
                # frame_outlet if not none
                if self.frame_outlet is not None:
                    self.frame_outlet(np.array([bit[1] for bit in self.bit_buffer]))

                for _ in range(self.bit_buffer.size()):
                    self.bit_buffer.dequeue()

                self.post_preamble = False
                self.signal_cursor = i
                

    fig = None
    def plot_signal(self):
        signal = np.array(self.sample_deque)

        if len(signal) == 0:
            return

        if self.fig is not None:
            plt.close(self.fig)

        self.fig = plt.figure()
        time_axis = np.arange(len(signal)) # / self.sampling_rate
        cursor_time = self.signal_cursor / self.sampling_rate  # Calculate time for cursor position

        plt.plot(time_axis, signal, color='orange', label='Signal')
        plt.axvline(x=cursor_time, color='blue', linestyle='--', label='Cursor Position')  # Plot vertical line at cursor

        plt.title("Synchronization Engine (" + ("post" if self.post_preamble else "pre") + "-preamble)")
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()  # Add legend to differentiate the signal and cursor line
        plt.show()


    def ensure_mem(self):
        diff = len(self.sample_deque) - self.samples_limit
        self.__remove_buffer(diff)

    def ensure_min(self):
        return len(self.sample_deque) >= self.min_samples

    def __remove_buffer(self, no_of_samples):
        self.signal_cursor -= no_of_samples
        if self.signal_cursor < 0:
            self.signal_cursor = 0
        for _ in range(no_of_samples):
            if self.sample_deque:
                self.sample_deque.popleft()


class FixedLengthQueue:
    def __init__(self, max_size):
        """Initialize the buffer with a maximum capacity."""
        self.max_size = max_size
        self.queue = []

    def enqueue(self, data):
        """Add data to the queue. If full, remove the oldest item (FIFO)."""
        if len(self.queue) == self.max_size:
            self.queue.pop(0)  # Remove the oldest item (FIFO)
        self.queue.append(data)

    def dequeue(self):
        """Remove and return the oldest element. Return None if the queue is empty."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def size(self):
        """Return the current size of the queue."""
        return len(self.queue)

    def is_full(self):
        """Check if the queue has reached its max capacity."""
        return len(self.queue) == self.max_size

    def __iter__(self):
        """Make the queue iterable."""
        return iter(self.queue)

    def __getitem__(self, index):
        """Allow indexing to access elements in the queue."""
        if 0 <= index < len(self.queue):
            return self.queue[index]
        else:
            raise IndexError("Index out of range")

    def __str__(self):
        """Return a string representation of the queue's contents."""
        return f"({', '.join(map(str, self.queue))})"



