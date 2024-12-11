# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "blue" ball in HSV
blueLower = (94, 80, 2)
blueUpper = (126, 255, 255)
pts = [deque(maxlen=args["buffer"]) for _ in range(3)]  # For 3 balls
centroids_prev = [None] * 3  # Store previous centroids of balls

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "blue", then clean it up
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centers = []  # Store centers for all detected balls

    # only proceed if at least one contour was found
    for c in cnts:
        # Filter out small contours
        if cv2.contourArea(c) < 500:
            continue

        # compute the minimum enclosing circle and centroid
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            centers.append(center)

    # Associate detected centroids with previous ones to avoid crossing paths
    for i, center in enumerate(centers):
        if centroids_prev[i] is None:
            centroids_prev[i] = center  # Initialize if this is the first frame
        else:
            # Calculate the distance between the current and previous centroids
            distances = [np.linalg.norm(np.array(center) - np.array(prev)) for prev in centroids_prev]
            closest_idx = np.argmin(distances)  # Find the closest previous centroid
            pts[closest_idx].appendleft(center)  # Update the corresponding trajectory
            centroids_prev[closest_idx] = center  # Update the previous centroid

    # Update tracked points for each ball
    for i in range(len(pts)):
        for j in range(1, len(pts[i])):
            if pts[i][j - 1] is None or pts[i][j] is None:
                continue
            thickness = int(np.sqrt(args["buffer"] / float(j + 1)) * 2.5)
            cv2.line(frame, pts[i][j - 1], pts[i][j], (255, 0, 0), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
