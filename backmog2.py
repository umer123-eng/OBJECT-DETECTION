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

# Function to apply background subtraction
def apply_background_subtraction(frame, background_subtractor):
    # Apply the background subtractor to the frame
    fg_mask = background_subtractor.apply(frame)
    
    return fg_mask

# Initialize background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Reading videos
capture = cv2.VideoCapture('ambulance.mkv')

while True:
    isTrue, frame = capture.read()

    # Check if the frame was read correctly
    if not isTrue:
        print("Failed to read the frame. Exiting...")
        break

    # Apply background subtraction
    fg_mask = apply_background_subtraction(frame, background_subtractor)

    # Resize frames
    frame_resized = rescaleframe(frame)
    fg_mask_resized = rescaleframe(fg_mask)

    # Display the frames
    cv2.imshow('Original Video', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Resized Original Video', frame_resized)
    cv2.imshow('Resized Foreground Mask', fg_mask_resized)

    # Break loop on 'd' key press
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
