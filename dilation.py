import cv2
import numpy as np

# Function to rescale frames
def rescaleframe(frame, scale=0.3):
    # images, videos and live video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Function to change resolution for live video
def changeRes(width, height):
    # live VIDEO
    capture.set(3, width)
    capture.set(4, height)

# Function to apply threshold
def apply_threshold(frame, threshold_value=127):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply the threshold
    _, thresholded_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY)
    
    return thresholded_frame

# Function to apply dilation
def apply_dilation(frame, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_frame = cv2.dilate(frame, kernel, iterations=iterations)
    return dilated_frame

# Reading videos
capture = cv2.VideoCapture('ambulance.mkv')

while True:
    isTrue, frame = capture.read()

    # Check if the frame was read correctly
    if not isTrue:
        print("Failed to read the frame. Exiting...")
        break

    # Apply threshold
    thresholded_frame = apply_threshold(frame)

    # Apply dilation
    dilated_frame = apply_dilation(thresholded_frame)

    # Resize frames
    frame_resized = rescaleframe(frame)
    thresholded_frame_resized = rescaleframe(thresholded_frame)
    dilated_frame_resized = rescaleframe(dilated_frame)

    # Display the frames
    cv2.imshow('Original Video', frame)
    cv2.imshow('Thresholded Video', thresholded_frame)
    cv2.imshow('Dilated Video', dilated_frame)
    cv2.imshow('Resized Original Video', frame_resized)
    cv2.imshow('Resized Thresholded Video', thresholded_frame_resized)
    cv2.imshow('Resized Dilated Video', dilated_frame_resized)

    # Break loop on 'd' key press
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
