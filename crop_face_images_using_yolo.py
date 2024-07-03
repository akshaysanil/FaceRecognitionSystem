import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import os


# Load the YOLO model


model = YOLO("/home/akshay/Downloads/yolov8n-face.pt")

# Initialize the tracker and annotators
# tracker = sv.ByteTrack()
# box_annotator = sv.BoundingBoxAnnotator()
# label_annotator = sv.LabelAnnotator()

def process_image(image_path: str, output_path: str) -> None:
    # Read the image
    
    filename = image_path.split('/')[-1]
    frame = cv2.imread(image_path)

    
    # Perform detection
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Update tracker with detections
    # detections = tracker.update_with_detections(detections)

    # Iterate over detected faces and perform face recognition
    for detection in detections.xyxy:
        x1, y1, x2, y2 = map(int, detection[:4])
        face = frame[y1:y2, x1:x2]
        print('old',face.shape)
        face = cv2.resize(face,(160,160))
        print('new',face.shape)
        cv2.imwrite(output_path+'/'+filename,face)
        
# give the path to the directory
# for dir in os.walk('facenet/dataset/unaligned_faces'):
for dir in os.walk('/home/akshay/Downloads/30newDatasetjun27/second_15_data/balanced'):
    try:
        if '.jpg' in dir[2][0]:         # unalign_path     #new_saving_path
            output_path = dir[0].replace('balanced','balancedFaceCrops') # need to create the dir before
            
            os.mkdir(output_path)
            for file in dir[2] :
                process_image(dir[0]+'/'+file , output_path) 
                
    except Exception as e:
        print(e)



