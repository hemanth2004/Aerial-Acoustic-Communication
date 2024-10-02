import time
import numpy as np

class realtime_process:
    def  __init__(self, process, outlet, deadline, sample_rate):
        self.process = process
        self.outlet = outlet
        self.deadline = deadline
        self.sample_rate = sample_rate
        self.samples = np.array([])

    def update(self, new_samples):

        self.samples = np.concatenate((self.samples, new_samples))  
        current_duration = len(self.samples) / self.sample_rate

        if current_duration >= self.deadline:
            num_samples_to_process = int(self.deadline * self.sample_rate)
            samples_to_process = self.samples[:num_samples_to_process]
            
            processed_data = self.process(samples_to_process)
            self.outlet(processed_data)            
            self.samples = self.samples[num_samples_to_process:]  

    def clear(self):
        self.samples = np.array([])

