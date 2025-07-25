import cv2

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

# Function to detect contours
def detect_contours(frame):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame
    contour_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)
    
    return contour_frame

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

    # Detect contours
    contour_frame = detect_contours(thresholded_frame)

    # Resize frames
    frame_resized = rescaleframe(frame)
    thresholded_frame_resized = rescaleframe(thresholded_frame)
    contour_frame_resized = rescaleframe(contour_frame)

    # Display the frames
    cv2.imshow('Original Video', frame)
    cv2.imshow('Thresholded Video', thresholded_frame)
    cv2.imshow('Contour Video', contour_frame)
    cv2.imshow('Resized Original Video', frame_resized)
    cv2.imshow('Resized Thresholded Video', thresholded_frame_resized)
    cv2.imshow('Resized Contour Video', contour_frame_resized)

    # Break loop on 'd' key press
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
