## Import the required libraries
from ultralytics import YOLO
import cv2
from utils.jsonify import *
from utils.draw import *

## Initiate model and image source
video_path = "rtsp://admin:admin@192.168.115.173:1945"
# pipeline = "rtspsrc location=rtsp://admin:admin@192.168.191.47:1935 latency=100 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! appsink drop=1"
# video_path = 0
class_names = open('/home/sri/BtechFinalYearProject/model/labels.txt', 'r').read().splitlines()
model = YOLO(r'/home/sri/Downloads/project/cv/best.pt')
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

## Weed Flag
curr = False
prev = False
flag = False

## Open JSON file for writing
json_file = open('/home/sri/BtechFinalYearProject/json_files/frames.json', 'w')

## Loop through the frames of the video
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    for result in results:
        bboxs = result.boxes.xyxy.cpu()
        confs = result.boxes.conf.cpu()
        clss = result.boxes.cls.cpu()
        for bbox,cnf,cs in zip(bboxs,confs,clss):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            conf = cnf
            class_name = class_names[int(cs)]
            ## Setting the confidence Threshold as 20%
            
            if cnf > 20:
                plot_one_box(bbox,frame,class_name,2)
                # cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                temp = True
                curr = temp
                if curr != prev:
                    flag = True
                    prev = curr
            else:
                temp = False
                curr = temp
                if curr != prev:
                    flag = True
                    prev = curr

    frame_to_json(frame, json_file,flag)  ## Convert the frame to base64 and write JSON data to file
    end = time.time()
    fps = end - start
    cv2.putText(frame,text = f'{fps}', org = (20,20),fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale= 1, color = (255, 0, 0), thickness= 2)
    cv2.imshow('Window',frame)
    if cv2.waitKey(25) == ord('q'):
        break

## Close the JSON file
json_file.close()

## Release the video capture object
cap.release()
cv2.destroyAllWindows()
