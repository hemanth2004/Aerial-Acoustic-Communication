import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class RealTimeFigureManager:
    def __init__(self):
        """Initialize the RealTimeFigureManager with a single figure."""
        self.master_fig, self.sub_axes = plt.subplots(4, 1, figsize=(12, 12))  

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)

        self.colors = [
            ['orange', 'blue'],   
            ['green', 'red'],     
            ['purple', 'cyan'],    
            ['brown', 'pink']      
        ]

        self.lines = [[ax.plot([], [], color=self.colors[i][j])[0] 
                         for j in range(2)] for i, ax in enumerate(self.sub_axes)]

        for i, ax in enumerate(self.sub_axes):
            ax.set_xlim(0, 1)  
            ax.set_ylim(-1, 1)  
            ax.grid(True)

        plt.ion()  # Enable interactive mode
        plt.show(block=False)  # Show the figure without blocking

    def update_figure(self, index, new_data_x, new_data_y):
        """Update a specific subplot with new x and y data.

        :param index: A tuple of (plot_index, signal_index) where:
                      plot_index (0 or 1) indicates which plot to update,
                      signal_index (0 through 3) indicates which signal to update.
        """
        plot_index, signal_index = index

        if not (0 <= signal_index < 4) or not (0 <= plot_index < 2):
            raise ValueError(f"Index out of range: {index}. Must be (0-1, 0-3).")

        line = self.lines[signal_index][plot_index]
        line.set_xdata(new_data_x)  
        line.set_ydata(new_data_y)  

        ax = self.sub_axes[signal_index]
        ax.relim()
        ax.autoscale_view()

        plt.pause(0.01)  # Pause to allow for the plot to update

    def plot_thread(self):
        """Run the plot in a separate thread."""
        while True:
            plt.pause(0.01)  # Keep the plot responsive

    def start_plotting(self):
        """Start the plotting thread."""
        threading.Thread(target=self.plot_thread, daemon=True).start()

if __name__ == "__main__":
    manager = RealTimeFigureManager()
    manager.start_plotting()  # Start the plotting in a separate thread

    for t in range(100):  
        new_x = np.linspace(0, 1, 500)
        new_y1 = np.sin(2 * np.pi * (t / 50) + new_x)  
        new_y2 = np.cos(2 * np.pi * (t / 50) + new_x)  

        manager.update_figure(index=(0, 0), new_data_x=new_x, new_data_y=new_y1)  
        manager.update_figure(index=(1, 0), new_data_x=new_x, new_data_y=new_y2)  

        if t % 10 == 0:  
            new_y3 = np.sin(2 * np.pi * (t / 50) + new_x + 1)  
            new_y4 = np.cos(2 * np.pi * (t / 50) + new_x + 1)  
            manager.update_figure(index=(0, 1), new_data_x=new_x, new_data_y=new_y3)  
            manager.update_figure(index=(1, 1), new_data_x=new_x, new_data_y=new_y4)  

    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Display the final plot
