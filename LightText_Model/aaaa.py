import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

# Function to create a figure and return the axis
def create_figure(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    line, = ax.plot([], [])  # Create an empty line

    # Position the figure
    fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    
    return line, ax, fig

# Function to update the plots
def update_plots(lines, axes, figures):
    for i, (line, ax) in enumerate(zip(lines, axes)):
        # Generate new data for the plot (e.g., a simple sine wave)
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data + i)  # Vary phase based on index
        line.set_data(x_data, y_data)
        
        ax.relim()  # Update limits
        ax.autoscale_view()  # Rescale the view
        figures[i].canvas.draw()  # Redraw the figure

    # Schedule the next update
    root.after(100, update_plots, lines, axes, figures)  # Update every 100 ms

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Define positions for each figure (top-left, top-right, bottom-left, bottom-right)
positions = [
    (0, 0),                           # Top-left
    (screen_width // 2, 0),          # Top-right
    (0, screen_height // 2),          # Bottom-left
    (screen_width // 2, screen_height // 2)  # Bottom-right
]

# Create figures and store lines and axes
lines = []
axes = []
figures = []

for pos in positions:
    line, ax, fig = create_figure(*pos)
    lines.append(line)
    axes.append(ax)
    figures.append(fig)

# Start the update loop
update_plots(lines, axes, figures)

# Keep the script running to display figures
root.mainloop()
