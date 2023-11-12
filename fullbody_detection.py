import cv2

# Load the cascade
full_body_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_fullbody.xml')

# Read the input image
image = cv2.imread('p2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect full bodies in the image
bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the bodies
for (x, y, w, h) in bodies:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output image
cv2.imwrite('output2.jpg',image)
# cv2.imshow('Full Body Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
