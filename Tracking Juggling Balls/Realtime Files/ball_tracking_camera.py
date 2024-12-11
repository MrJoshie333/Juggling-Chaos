import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the white color range (for white objects like tennis balls)
whiteLower = (0, 0, 200)
whiteUpper = (180, 50, 255)

# Initialize trails for each ball
trails = [[], [], []]  # Separate trail lists for up to 3 balls
max_trail_length = 15  # 0.5 seconds at 30 fps
distance_threshold = 100  # Maximum distance for a point to belong to the same trail

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

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

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

    # Update each trail
    for i in range(len(trails)):
        if i < len(current_positions):
            if len(trails[i]) > 0:
                # Ensure the new point is within the distance threshold
                last_point = trails[i][-1]
                distance = np.sqrt((current_positions[i][0] - last_point[0]) ** 2 +
                                   (current_positions[i][1] - last_point[1]) ** 2)
                if distance < distance_threshold:
                    # Interpolate between the last point and the new position
                    interpolated_points = interpolate_points(last_point, current_positions[i])
                    trails[i].extend(interpolated_points)
                else:
                    # If too far, treat it as a new trail
                    trails[i] = [current_positions[i]]
            else:
                # If the trail is empty, start a new one
                trails[i].append(current_positions[i])
        else:
            # If no current position, clear the trail
            trails[i] = []

        # Limit the trail length
        if len(trails[i]) > max_trail_length:
            trails[i] = trails[i][-max_trail_length:]

        # Draw the trail
        for j in range(1, len(trails[i])):
            cv2.line(frame, trails[i][j - 1], trails[i][j], (0, 255, 0), 2)

        # Draw the current position of the ball
        if len(trails[i]) > 0:
            cx, cy = trails[i][-1]
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Tracking Balls with Smooth Trajectories', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
