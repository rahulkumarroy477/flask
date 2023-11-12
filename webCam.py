from flask import Flask, make_response, render_template, Response
from ultralytics import YOLO
import cv2 as cv
import time
from collections import Counter
app = Flask(__name__)
# Distance constants 
KNOWN_DISTANCE = 60 #CM
PERSON_WIDTH = 46 #CM
# MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX


shirt_sizes = {
    (42, 45): 'S',
    (45, 48): 'M',
    (48, 50): 'L',
    (50, 52): 'XL',
    (52, 60) : 'XXL'
}

# Number of frames to consider for moving average
output_stream = []

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
print(class_names)
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# Predicted person width in the frame
def predicted_person_width(focal_length, known_distance, width_in_frame):
    predicted_width = (width_in_frame * known_distance) / focal_length
    return predicted_width

ref_person = cv.imread('images/rimg1.png')

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
cap = cv.VideoCapture(0)
overlay = cv.imread('images/person-icon.png',cv.IMREAD_UNCHANGED)
width = int(cap.get(3))
height = int(cap.get(4))
size = ''
def generate_frames():
    start_time = time.time()
    while time.time()-start_time<20:
        ret, frame = cap.read()

        # Draw a vertical line in the middle of the window from top to bottom
        cv.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 1)
        data = object_detector(frame) 
        for d in data:
            if d[0] =='person':
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                predicted_width = predicted_person_width(focal_person, KNOWN_DISTANCE, d[1])
                for length_range, s in shirt_sizes.items():
                    if length_range[0] <= predicted_width < length_range[1]:
                        print(f"For a shirt with length {predicted_width}, the size is: {s}")
                        output_stream.append(s)
                        break
                    else:
                        print(f"Align properly")
                x, y = d[2]
                
            cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
            cv.putText(frame, f'Distance: {round(distance, 2)} cm, Width: {round(predicted_width, 2)} cm', (x+5, y+13), FONTS, 0.48, GREEN, 2)
        
        opacity = 0.5
        # Resize the overlay image to match the size of the frame
        overlay_resized = cv.resize(overlay, (frame.shape[1], frame.shape[0]))

        # Blend the images using cv2.addWeighted()
        blended_image = cv.addWeighted(frame, 1 - opacity, overlay_resized[:, :, :3], opacity, 0)

        cv.imshow('Blended Image', blended_image)

        print(f"Size of person: {size}")
                
        ret, buffer = cv.imencode('.jpg', blended_image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    
    size_counts = Counter(output_stream)

    print(size_counts)

    most_common_size = max(size_counts, key=size_counts.get, default=None)
    print(most_common_size)

@app.route('/')
def index():
    return render_template('popup.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
