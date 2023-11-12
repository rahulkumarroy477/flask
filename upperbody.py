import cv2

# Load the pre-trained upper body classifier
upper_body_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_upperbody.xml')

# Initialize the webcam (change the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect upper bodies in the frame
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected upper bodies
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the detected upper bodies
    cv2.imshow('T-shirt Boundary Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
