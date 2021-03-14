import numpy as np
import imutils
import cv2
import time
#load
prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
#confidence level
confThresh = 0.2
#initialize classes
CLASSES = ["background", "airplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
           "dog", "horse", "motorbike", "person", "plant", "sheep",
           "sofa", "train", "television"]
#initialize colors
COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))
#loading model
print("Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
#setup cam
print("Starting Camera Feed...")
cam = cv2.VideoCapture(1)
time.sleep(2.0)

while True:
    _,frame = cam.read()
    #resize frame display in window
    frame = imutils.resize(frame, width = 500)
    #get h and w of the frame
    (h,w = frame.shape[:2]
     #pre-process image pre-requisite size for mobileNetSSD
     imResize = cv2.resize(frame, (300, 300))
     #convert image into blob
     blob = cv2.dnn.blobFromImage(imResize, 0.007843, (300, 300), 127.5)
     #pass network image
     net.setInput(blob)
     #get accuracy, classID, coordinate of the rectangle
     detections = net.forward()
     #detect shape
     detectShape = detections.shape[2]
     #loop to detect some kind of shapes
     for i in np.arange(0, detectShape):
         #i = multiple objects, 
         confidence = detections[0, 0, i, 2]
         #confidence level
         if confidence > confTresh:
             #obtain CLASSES id
             idx = int(detections[0, 0, i, 1])
             #ref.
             print("ClassID: ", detections[0, 0, i, 1])
             '''
             certain box 3:7 = return value of detection:
                position: start x,y and end x,y
                and convert to array (np.array)
             '''
             box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
             #get the int
             (startX, startY, endX, endY) = box.astype("int")
            
             label = "{}: {:.2f}%".format(CLASSES[idx],
                                          confidence * 100)
             #draw rectangle in objects
             cv2.rectangle(frame, (startX, startY), (endX, endY),
                           COLORS[idx], 2)
             #putting text above the detected objects
             if startY - 15 > 15:
                     y = startY - 15
             else:
                 startY + 15
                 cv2.putText(frame, label, (startX, y),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             COLORS[idx], 2)

        cv2.imshow("Object Recognition", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cam.release()
cv2.destroyAllWindows()
     
     

