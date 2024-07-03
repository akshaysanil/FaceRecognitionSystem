import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import *
import cv2 as cv
from datetime import datetime


model = YOLO("/home/akshay/Downloads/yolov8n-face.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = set()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]



    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    
    facenet_results = []  # Initialize a list to collect face recognition results
    
    for detection,label in zip(detections.xyxy,labels):
        x1, y1, x2, y2 = map(int, detection[:4])
        
        face = frame[y1:y2, x1:x2]
        print('getting face')
        facenet_result = predict_face(face)
        name = facenet_result
        
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename = f'{name}_{timestamp}.jpg'
        print('filename : ',filename)
        output_dir = '/home/akshay/work/mysur_tests/machineTest1/facenet_new/marked_attendance'
        filepath = os.path.join(output_dir, filename)

        # saving faceimage with timestamp 
        # if name not in saved_names:
        #     cv.imwrite(filepath,face)
        #     saved_names.add(name)

        facenet_results.append(facenet_result)[0]


    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels , text_from_facenet = facenet_results)

sv.process_video(
    source_path="output.mp4",
    target_path="result2.mp4",
    callback=callback
)


# ------------------------------- core file edited in this path -------------------------------------------
# /home/akshay/anaconda3/envs/py38/lib/python3.8/site-packages/supervision/annotators/core.py