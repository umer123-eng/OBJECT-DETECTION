import cv2
import numpy as np
import math

# Function to rescale frames
def rescaleframe(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Reading videos
capture = cv2.VideoCapture('ambulance.mkv')

# Get the frame rate (frames per second)
fps = capture.get(cv2.CAP_PROP_FPS)

# Create a background object
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
kernel = np.ones((3, 3), np.uint8)
kernel2 = None

# Dictionary to store object positions (use contour index as key)
object_positions = {}

while True:
    ret, frame = capture.read()
    if not ret:
        break

    fgmask = backgroundObject.apply(frame)
    __, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=6)

    # Detect the contours
    contours, __ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    # Loop through the contours and track large objects
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 20000:
            # Get the bounding box coordinates
            x, y, width, height = cv2.boundingRect(cnt)
            # Draw a rectangle around the object
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 255, 0), thickness=5)

            # Calculate the object's center
            center = (x + width // 2, y + height // 2)

            # Track object by storing the position of its center using the index 'i' as the key
            if i in object_positions:
                # If object already tracked, calculate the distance moved
                previous_center = object_positions[i]
                distance = calculate_distance(previous_center, center)
                speed = (distance / fps) * 3.6  # Convert from pixels/frame to km/h
                cv2.putText(frameCopy, f"Speed: {speed:.2f} km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            # Update the object's current position
            object_positions[i] = center

            # Write a text near the object
            cv2.putText(frameCopy, "OBJECT DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    foreground = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Resize frames before displaying
    resized_frame = rescaleframe(frame, scale=0.4)
    resized_foreground = rescaleframe(foreground, scale=0.4)
    resized_frameCopy = rescaleframe(frameCopy, scale=0.4)
    resized_fgmask = rescaleframe(fgmask, scale=0.4)

    cv2.imshow('Foreground', resized_foreground)
    cv2.imshow('Detected frame', resized_frameCopy)
    cv2.imshow('FG Mask', resized_fgmask)
    cv2.imshow('Original Frame', resized_frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
