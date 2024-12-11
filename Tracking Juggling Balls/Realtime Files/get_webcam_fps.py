import cv2
import time

# Initialize webcam with DirectShow
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera


# Set the desired resolution and FPS
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
# cap.set(cv2.CAP_PROP_FPS, 60)

# Set MJPEG codec (MJPG)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Measure time between frames
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    print(f"Time per frame: {time_diff:.6f} seconds")  # Time between frames
    print(f"FPS: {1/time_diff:.2f}")

    # Show live feed
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
