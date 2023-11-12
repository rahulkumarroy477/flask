import cv2
import numpy as np

webcam = False
path = 'p1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,980)
cap.set(4,720)

while True:
    if webcam:success,img = cap.read()
    else:img = cv2.imread(path)
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()