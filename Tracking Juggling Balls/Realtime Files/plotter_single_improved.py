import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pygetwindow as gw
import time

# Start timing
start_time = time.time()

print("Initializing webcam...")
# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print(f"Webcam initialized in {time.time() - start_time:.2f} seconds.")

# Set the camera to use MJPEG format
print("Configuring webcam settings...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verify resolution settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to {int(width)}x{int(height)}")

# Define the white color range (for white objects like tennis balls)
whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

# Initialize lists to store x and y positions of the ball
x_positions = []
y_positions = []

# Create a blank canvas (change dimensions to your desired resolution)
canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Set up the matplotlib figure
plt.ion()  # Interactive mode for live plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, canvas_width)
ax.set_ylim(canvas_height, 0)  # Invert y-axis to match webcam view
line, = ax.plot([], [], 'ro-', label="Ball Trajectory")
ax.set_xlabel('X Position (pixels)')
ax.set_ylabel('Y Position (pixels)')
ax.set_title('Ball Trajectory Over Time')
ax.legend()
plt.show()

# Button to toggle points on/off
def toggle_points(event):
    global show_points
    show_points = not show_points
    line.set_marker('o' if show_points else '')
    fig.canvas.draw()

ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # Positioning of button
button = Button(ax_button, 'Toggle Points')
button.on_clicked(toggle_points)

# Display the canvas with the updated ball position
frame_count = 0
show_points = True  # Initially, show points on the trajectory
loop_start_time = time.time()

print("Starting main loop...")

# Main loop to track and plot the ball
while True:
    # Measure frame capture time
    frame_start_time = time.time()

    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_count += 1
    if frame_count == 1:
        print(f"First frame captured after {time.time() - loop_start_time:.2f} seconds.")

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for white objects
    mask = cv2.inRange(hsv, whiteLower, whiteUpper)

    # Apply a Gaussian blur to the mask to smooth it out
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Find contours (balls) in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_position = None
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum size to filter noise
            # Get the center of the contour (ball position)
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                current_position = (cx, cy)

    if current_position:
        cx, cy = current_position

        # Record x and y positions
        x_positions.append(cx)
        y_positions.append(cy)

        # Draw the ball on the canvas (this can be shown in the live feed if you wish)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    # Display the canvas with the updated ball position
    cv2.imshow('Tracking Ball', frame)

    # Update the plot with new data
    line.set_data(x_positions, y_positions)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Measure frame processing time
    frame_time = time.time() - frame_start_time
    if frame_count % 30 == 0:  # Every 30 frames (~1 second)
        print(f"Processed frame {frame_count}. Last frame time: {frame_time:.2f} seconds.")

    # Try to ensure the windows are on monitor 2 (left monitor)
    # Get the window of the webcam
    windows = gw.getWindowsWithTitle("Tracking Ball")
    if windows:
        win = windows[0]
        win.activate()  # Bring the window to the front
        # Positioning the webcam window on monitor 2 (left monitor)
        win.moveTo(-1920, 0)  # Assuming monitor 2 is to the left of the primary monitor

    # Get the matplotlib window (plot)
    plt_windows = gw.getWindowsWithTitle("matplotlib")
    if plt_windows:
        plt_win = plt_windows[0]
        plt_win.activate()  # Bring the window to the front
        # Positioning the matplotlib window on monitor 2 (left monitor)
        plt_win.moveTo(-1920, 0)  # Adjust for left monitor position

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Save the plot after closing the loop
plt.ioff()  # Turn off interactive mode
plt.savefig('ball_trajectory.png')
plt.show()

print("Webcam released. Plot saved as 'ball_trajectory.png'.")
