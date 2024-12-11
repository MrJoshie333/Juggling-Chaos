import cv2
import numpy as np
import time

# Set webcam properties
FRAME_WIDTH = 640  # 720p
FRAME_HEIGHT = 360
FPS = 60  # Match playback and recording speed

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Video writer for raw video (real-time playback)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('juggling_trajectory_raw.avi', fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

# Initialize tracking
recording_start_time = time.time()
trajectories = {1: [], 2: [], 3: []}  # For three balls

def detect_white_balls(frame):
    """Detect white balls in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define range for white color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

recording_done = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If recording, save the raw frame
    if not recording_done:
        out.write(frame)
        if time.time() - recording_start_time >= 10:  # Stop recording after 10 seconds
            recording_done = True
            print("Recording complete. Video saved as 'juggling_trajectory_raw.avi'.")

    # Process frame to detect white balls
    mask = detect_white_balls(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_positions = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum size to filter noise
            # Get the center of the ball
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Add to the current positions if radius is within reasonable range
            if 10 < radius < 50:  # Adjust these limits based on your balls' size
                current_positions.append(center)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

    # Manage trajectories
    for idx, position in enumerate(current_positions):
        if idx < 3:  # Max three balls
            trajectories[idx + 1].append(position)

    # Draw trajectories
    for key, traj in trajectories.items():
        for i in range(1, len(traj)):
            cv2.line(frame, traj[i - 1], traj[i], (255, 0, 0), 2)

    # Show live feed with trajectories
    cv2.imshow('Juggling Tracking', frame)

    # Stop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
