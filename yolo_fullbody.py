from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
# Load YOLO model
model = YOLO('yolov8n.pt')

while True:
    success,frame = cap.read()
    results = model(frame,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
    cv2.imshow("Image",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

