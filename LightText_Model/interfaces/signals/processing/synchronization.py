import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

class sync_ask:
    # engine division into pre preamble phase and post-preamble phase
    post_preamble = False
    engine_on = False

    sampling_rate = 44100
    min_samples = sampling_rate * 0.5
    preamble = np.array([])
    samples_limit = sampling_rate * 16 # seconds


    sample_deque = deque()
    bit_list = np.array([])

    # Tracking transitions and timing
    edge_list = []  # Store indices of detected edges
    window_index = -1  # Index of the sample indicating the start of a symbol
    transition_intervals = []  # To track time between transitions
    first_edge = -1

    processed_in_this_frame = 0
    fixed_frame_size = None

    def __init__(self, demod, sampling_rate, symbol_duration, preamble, bit_outlet=None, frame_outlet=None, plot=True, bottleneck_deadline=2, fixed_frame_size=None):
        self.demod = demod # the demod function that. dont worry about it for now though
        self.sampling_rate = sampling_rate
        self.symbol_duration = symbol_duration
        self.bit_outlet = bit_outlet
        self.frame_outlet = frame_outlet
        self.preamble = preamble
        self.plot = plot
        self.bottleneck_deadline = bottleneck_deadline
        self.fixed_frame_size = fixed_frame_size

    def __remove_buffer(self, no_of_samples):
        diff = no_of_samples
        if diff > 0:
            for i in range(diff):
                # Remove the oldest sample from the deque
                self.sample_deque.popleft()
                self.window_index -= 1 if self.window_index >= 0 else 0
                self.first_edge -= 1 if self.first_edge >= 0 else 0
                self.end_of_frame -= 1 if self.end_of_frame >= 0 else 0
                
                # Update edge_list to remove edges that are no longer valid
                if self.edge_list:
                    # Remove edges that are now out of bounds
                    self.edge_list = [edge for edge in self.edge_list if edge >= 0]
                    # Adjust indices in edge_list by subtracting one for each removed sample
                    self.edge_list = [edge - 1 for edge in self.edge_list if edge - 1 >= 0]

    def set_engine_status(self, status):
        if status:
            self.engine_on = True
        else:
            if not self.engine_on:
                return
            
            t = 0
            if self.post_preamble and self.window_index > 0 and self.first_edge > 0:
                rem = self.window_index - self.first_edge
                max = self.fixed_frame_size
                rem = max - rem
                t = abs(rem / self.sampling_rate)
                print("time left: ", t)
            else:
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
            self.pre_preamble_sync()
            self.post_preamble = self.detect_preamble()
            if self.post_preamble:
                # print("Entering Post-Preamble Phase")
                self.processed_in_this_frame = 0  # Reset symbol counter
                self.process_symbols_post_preamble()
        else:  # Post-preamble phase: process frame symbols
            self.process_symbols_post_preamble()

        if self.plot:
            self.plot_signal()

    fig = None
    def plot_signal(self):
        # Convert deque to numpy array for plotting
        signal = np.array(self.sample_deque)

        if len(signal) == 0:
            return

        # Create a time axis based on the number of samples and sampling rate
        time_axis = np.arange(len(signal)) / self.sampling_rate

        # Close the previous figure if it exists
        if self.fig is not None:
            plt.close(self.fig)

        # Create a new figure
        self.fig = plt.figure()

        # Clear the current axes
        plt.clf()

        if 0 <= self.end_of_frame < len(signal):
            # Part 1: Before window_index (greyed out)
            plt.plot(time_axis[:self.end_of_frame], signal[:self.end_of_frame],
                    label='Sample Values (Before Window)', color='gray', alpha=0.75)

            # Part 2: From window_index to the end (regular style)
            plt.plot(time_axis[self.end_of_frame:], signal[self.end_of_frame:],
                    label='Sample Values (After Window)', color='blue')
        else:
            # If `window_index` is out of bounds, plot the entire signal normally
            plt.plot(time_axis, signal, label='Sample Values')

        # Calculate the time corresponding to the window start index
        if 0 <= self.window_index < len(signal):
            window_time = self.window_index / self.sampling_rate
            plt.axvline(x=window_time, color='r', linestyle='--', label='Window Start')

        # Optionally mark the detected edges using time values
        # for edge in self.edge_list:
        #     if 0 <= edge < len(signal):
        #         edge_time = edge / self.sampling_rate
        #         plt.axvline(x=edge_time, color='g', linestyle=':')

        # Plot periodic lines starting from `first_edge` at intervals of `symbol_duration`
        if self.first_edge >= 0:
            symbol_interval = int(self.symbol_duration * self.sampling_rate)  # Calculate symbol interval in samples
            for position in range(self.first_edge, len(signal), symbol_interval):
                line_time = position / self.sampling_rate
                plt.axvline(x=line_time, color='b',linewidth=0.5, linestyle='--', alpha=1)  # Blue dashed lines for symbol intervals

        # Add labels and legend
        plt.title('Sync Engine Visualisation ({mode})[{status}]'.format(
            mode="post-preamble" if self.post_preamble else "pre-preamble",
            status="ON" if self.engine_on else "OFF"
        ))
        plt.xlabel('Time (s)')  # Time axis in seconds
        plt.ylabel('Amplitude')
        plt.legend()

        # Set x-ticks to have more granularity
        num_ticks = 10  # Increase the number of ticks
        max_time = len(signal) / self.sampling_rate  # Total time duration in seconds
        tick_spacing = max_time / num_ticks  # Compute the spacing between ticks

        # Generate tick positions and labels
        tick_positions = np.arange(0, max_time + tick_spacing, tick_spacing)
        plt.xticks(tick_positions)  # Set custom tick positions

        # Enable minor ticks for even finer granularity
        plt.minorticks_on()
        plt.tick_params(axis='x', which='minor', length=5, color='gray', direction='in')

        plt.show(block=False)  # Ensure the plot stays open

    def pre_preamble_sync(self):
        signal = np.array(self.sample_deque)

        # Find edges (1 -> 0 or 0 -> 1)
        for i in range(1, len(signal)):
            if signal[i] != signal[i - 1]:  # Edge detected
                self.edge_list.append(i)

                # Calculate time between this edge and the previous one
                if len(self.edge_list) > 1:
                    interval = (self.edge_list[-1] - self.edge_list[-2]) / self.sampling_rate
                    self.transition_intervals.append(interval)

                # If we have enough edges, reposition the window to match the pattern
                # if len(self.edge_list) >= 2:
                #     self.window_index = self.edge_list[-1] - int(self.symbol_duration * self.sampling_rate / 2)

    def process_symbols_post_preamble(self):
        """Process symbols until the frame is complete."""
        total_frame_samples = int(self.fixed_frame_size * self.symbol_duration * self.sampling_rate)

        # Process symbols until the frame size is reached
        while (len(self.sample_deque) - self.window_index) >= self.min_samples and \
            self.processed_in_this_frame < self.fixed_frame_size:
            self.process_symbol()

        # If we've processed the expected number of symbols, reset to pre-preamble phase
        if self.processed_in_this_frame >= self.fixed_frame_size:
            print("Frame completed. Returning to pre-preamble phase.")

            self.end_of_frame = self.window_index + int(self.fixed_frame_size * self.symbol_duration * self.sampling_rate)
        
            self.post_preamble = False

            if self.frame_outlet is not None:
                self.frame_outlet(self.frame_bits)
            self.frame_bits = np.array([])

            self.processed_in_this_frame = 0  # Reset for next frame
            self.window_index = -1

    frame_bits = np.array([])
    def process_symbol(self):
        if self.window_index < 0:
            return
        self.window_index += int(self.symbol_duration * self.sampling_rate)
        bit = self.sample_deque[self.window_index]

        if self.bit_outlet is not None:
            self.bit_outlet(bit)
        self.frame_bits = np.append(self.frame_bits, [bit])
        self.processed_in_this_frame += 1
        return bit

    end_of_frame = -1
    def detect_preamble(self):
        signal = np.array(self.sample_deque)
        length_threshold = int(len(self.preamble) * self.symbol_duration * self.sampling_rate)

        # Define tolerance limits
        min_length = int(0.75 * length_threshold)
        max_length = int(1.2 * length_threshold)

        continuous_ones = 0
        max_continuous_ones = 0  # Track the longest contiguous block of 1s
        interruptions = 0
        found_valid_sequence = False

        # New variable to track the end of the detected preamble block
        end_of_preamble_block = -1

        # Start the loop from `end_of_frame + 1` to avoid re-checking previous data
        for i, sample in enumerate(signal):
            if i <= self.end_of_frame:
                continue  # Skip all samples before the end of the previous frame

            if sample == 1:
                continuous_ones += 1
                interruptions = 0  # Reset interruptions since we found a 1

                if continuous_ones > max_continuous_ones:
                    max_continuous_ones = continuous_ones

                # Check if we are within valid length range
                if min_length <= continuous_ones <= max_length:
                    found_valid_sequence = True
                    end_of_preamble_block = i  # Use `end_of_preamble_block` for the position where the block ends

                    # Stop searching once the first valid block is found
                    break  # <-- Ensure only the first valid preamble is chosen
            else:
                interruptions += 1

                # Allow for a limited number of interruptions
                if interruptions > self.sampling_rate * 0.2:
                    continuous_ones = 0  # Reset the counter
                    interruptions = 0  # Reset interruptions count

        result = found_valid_sequence and max_continuous_ones >= min_length
        if result:
            # Use `end_of_preamble_block` for window positioning
            self.window_index = end_of_preamble_block + int((self.symbol_duration * self.sampling_rate / 2.0) * 0.85)
            self.first_edge = self.window_index
        return result


    def ensure_mem(self):
        diff = len(self.sample_deque) - self.samples_limit
        self.__remove_buffer(diff)

    def ensure_min(self):
        return len(self.sample_deque) >= self.min_samples


    

# example array of input signal chunk
# symbol duration = 1s
# sampling_rate = 12
# input_signal = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0] // incomplete ups and downs