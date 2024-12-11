import cv2
import numpy as np
import matplotlib.pyplot as plt
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
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verify resolution settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to {int(width)}x{int(height)}")

# Define the white color range (for white objects like tennis balls)
whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

# Initialize trails for each ball
trails = [[], [], []]  # Separate trail lists for up to 3 balls
y_positions = [[] for _ in range(3)]  # List to store y-values for up to 3 balls
max_trail_length = 15  # 0.5 seconds at 30 fps
distance_threshold = 25  # Maximum distance for a point to belong to the same trail
time_step = 0.033  # Approx. 30 fps, constant time step in seconds

# Create a blank canvas (change dimensions to your desired resolution)
canvas_height, canvas_width = 480, 640
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

print("Starting main loop...")

def interpolate_points(p1, p2, steps=5):
    """
    Interpolate between two points with a specified number of steps.
    Args:
        p1 (tuple): Starting point (x1, y1).
        p2 (tuple): Ending point (x2, y2).
        steps (int): Number of interpolated points.
    Returns:
        list: List of interpolated points.
    """
    x_vals = np.linspace(p1[0], p2[0], steps).astype(int)
    y_vals = np.linspace(p1[1], p2[1], steps).astype(int)
    return [(x, y) for x, y in zip(x_vals, y_vals)]

frame_count = 0
loop_start_time = time.time()

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

    # Sort contours based on area and only keep the largest 3
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Loop through contours and update trails
    current_positions = []
    for i, contour in enumerate(contours):
        # Get the center of the contour (ball position)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_positions.append((cx, cy))

    # Clear the canvas each frame (comment this line if you want trails to persist indefinitely)
    canvas.fill(0)

    # Update each trail
    for i in range(len(trails)):
        if i < len(current_positions):
            cx, cy = current_positions[i]

            # Record y-position
            y_positions[i].append(cy)

            if len(trails[i]) > 0:
                # Ensure the new point is within the distance threshold
                last_point = trails[i][-1]
                distance = np.sqrt((cx - last_point[0]) ** 2 + (cy - last_point[1]) ** 2)
                if distance < distance_threshold:
                    # Interpolate between the last point and the new position
                    interpolated_points = interpolate_points(last_point, (cx, cy))
                    trails[i].extend(interpolated_points)
                else:
                    # If too far, treat it as a new trail
                    trails[i] = [(cx, cy)]
            else:
                # If the trail is empty, start a new one
                trails[i].append((cx, cy))
        else:
            # If no current position, clear the trail
            trails[i] = []

        # Limit the trail length
        if len(trails[i]) > max_trail_length:
            trails[i] = trails[i][-max_trail_length:]

        # Draw the trail
        for j in range(1, len(trails[i])):
            cv2.line(canvas, trails[i][j - 1], trails[i][j], (0, 255, 0), 2)

        # Draw the current position of the ball
        if len(trails[i]) > 0:
            cv2.circle(canvas, trails[i][-1], 10, (0, 255, 0), -1)

    # Display the canvas
    cv2.imshow('Tracking Balls with Smooth Trajectories', canvas)

    # Measure frame processing time
    frame_time = time.time() - frame_start_time
    if frame_count % 30 == 0:  # Every 30 frames (~1 second)
        print(f"Processed frame {frame_count}. Last frame time: {frame_time:.2f} seconds.")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Processing data...")

# Create time steps
num_points = max(len(y) for y in y_positions)  # Longest recorded trajectory
time_steps = np.arange(0, num_points * time_step, time_step)

# Plot y-positions
plt.figure(figsize=(10, 6))
for i, y_vals in enumerate(y_positions):
    if y_vals:
        plt.plot(time_steps[:len(y_vals)], y_vals, label=f'Ball {i + 1}')
plt.xlabel('Time (s)')
plt.ylabel('y-Position (pixels)')
plt.title('y-Position of Balls Over Time')
plt.legend()
plt.grid()
plt.savefig('ball_y_positions_with_feedback.png')
plt.show()

print("Plot saved as 'ball_y_positions_with_feedback.png'.")
