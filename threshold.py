import cv2
import numpy as np

# Function to rescale frames
def rescaleframe(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Function to calculate the Euclidean distance
def calculate_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

# Reading the video
capture = cv2.VideoCapture('ambulance.mkv')

# Create a background subtractor object
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=False)
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

# Dictionary to store object positions across frames
object_positions = {}
frame_count = 0
fps = capture.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_count += 1

    # Apply the background subtractor
    fgmask = backgroundObject.apply(frame)

    # Threshold and clean up the mask
    __, fgmask = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel2, iterations=4)

    # Detect contours
    contours, __ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    current_frame_positions = []  # Store positions for the current frame

    # Loop through contours and detect moving objects (cars)
    for cnt in contours:
        if cv2.contourArea(cnt) > 5000:
            x, y, width, height = cv2.boundingRect(cnt)
            center = (x + width // 2, y + height // 2)
            current_frame_positions.append(center)

            # Draw rectangle around the detected object
            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 255, 0), thickness=2)
            cv2.putText(frameCopy, "OBJECT DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Calculate speed for each object by matching with previous positions
    for i, current_position in enumerate(current_frame_positions):
        if i in object_positions:
            prev_position = object_positions[i]
            distance_pixels = calculate_distance(prev_position, current_position)
            # Convert pixel distance to real-world units if needed
            speed = (distance_pixels * fps) / 30  # Assuming 30 pixels = 1 unit (e.g., meter)

            # Display speed on the frame
            cv2.putText(frameCopy, f"Speed: {speed:.2f} units/s", (current_position[0] + 10, current_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

        # Update the current position for the object
        object_positions[i] = current_position

    # Resize frames for display
    resized_frame = rescaleframe(frame, scale=0.6)
    resized_frameCopy = rescaleframe(frameCopy, scale=0.6)
    resized_fgmask = rescaleframe(fgmask, scale=0.6)

    # Display frames
    cv2.imshow('Detected Cars', resized_frameCopy)
    cv2.imshow('FG Mask', resized_fgmask)
    cv2.imshow('Original Frame', resized_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close all windows
capture.release()
cv2.destroyAllWindows()
