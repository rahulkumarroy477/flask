from flask import Flask, make_response, render_template, Response
from ultralytics import YOLO
import cv2

global camera
app = Flask(__name__)
camera = cv2.VideoCapture(0)

model = YOLO('yolov8n.pt')

def generate_frames():
    while True:

        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame,stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                    print(x1,y1,x2,y2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('popup.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
