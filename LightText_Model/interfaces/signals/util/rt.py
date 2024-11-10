import numpy as np

class realtime_process:
    def __init__(self, process, outlet, deadline, sample_rate, signals=1):
        self.process = process
        self.outlet = outlet
        self.deadline = deadline
        self.sample_rate = sample_rate
        self.samples = np.array([]) if signals == 1 else [np.array([]) for _ in range(signals)]
        self.signals = signals

    def update(self, new_samples):
        if self.signals > 1:
            assert len(new_samples) == self.signals
            for i in range(self.signals):
                self.samples[i] = np.concatenate((self.samples[i], new_samples[i]))
            current_duration = len(self.samples[0]) / self.sample_rate
        else:
            self.samples = np.concatenate((self.samples, new_samples))
            current_duration = len(self.samples) / self.sample_rate

        if current_duration >= self.deadline:
            num_samples_to_process = int(self.deadline * self.sample_rate)
            
            if self.signals > 1:
                # print("found a ", self.signals, " signal process")
                samples_to_process = [self.samples[i][:num_samples_to_process] for i in range(self.signals)]
                processed_data = self.process(samples_to_process)
                for i in range(self.signals):
                    self.samples[i] = self.samples[i][num_samples_to_process:]
                self.outlet(processed_data)
            else:
                samples_to_process = self.samples[:num_samples_to_process]
                processed_data = self.process(samples_to_process)
                self.outlet(processed_data)
                self.samples = self.samples[num_samples_to_process:]

    def clear(self):
        if self.signals > 1:
            self.samples = [np.array([]) for _ in range(self.signals)]
        else:
            self.samples = np.array([])
