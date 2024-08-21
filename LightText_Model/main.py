import time
import os

def clear_terminal():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

def update(delta_time):
    # Your game or application logic goes here
    pass

def main():
    max_delta_time = 0.1  # Maximum delta time in seconds
    previous_time = time.time()
    fps = 0

    while True:
        current_time = time.time()
        delta_time = current_time - previous_time
        previous_time = current_time

        # Cap the delta time to the maximum
        if delta_time > max_delta_time:
            delta_time = max_delta_time

        # Ensure delta_time is not zero to avoid division by zero
        if delta_time == 0:
            delta_time = 1e-6  # Set to a very small value

        # Call the update function with the capped delta time
        update(delta_time)

        # Calculate FPS
        fps = 1.0 / delta_time

        # Clear the terminal and display FPS
        clear_terminal()
        print(f"FPS: {fps:.2f}")

        # Sleep for a short time to prevent the loop from running too fast
        time.sleep(0.01)

if __name__ == "__main__":
    main()
