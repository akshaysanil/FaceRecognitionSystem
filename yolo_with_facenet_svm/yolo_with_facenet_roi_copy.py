import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import *
import cv2 as cv
from datetime import datetime
import os

# Define your YOLO model and trackers
model = YOLO("/home/akshay/Downloads/yolov8n-face.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = set()

# ROI coordinates
roi_x1, roi_y1, roi_x2, roi_y2 = 450, 200, 1200, 1200  # Example coordinates

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Draw the ROI on the frame
    cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = []
    facenet_results = []  # Initialize a list to collect face recognition results

    for detection, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
        x1, y1, x2, y2 = map(int, detection[:4])

        # Check if the bounding box is completely within the ROI
        if roi_x1 <= x1 <= roi_x2 and roi_x1 <= x2 <= roi_x2 and roi_y1 <= y1 <= roi_y2 and roi_y1 <= y2 <= roi_y2:
            labels.append(f"#{tracker_id} {results.names[class_id]}")

            face = frame[y1:y2, x1:x2]
            print('getting face')
            facenet_result = predict_face(face)
            name = facenet_result

            timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
            filename = f'{name}_{timestamp}.jpg'
            print('filename : ', filename)
            output_dir = '/home/akshay/work/mysur_tests/machineTest1/facenet_new/marked_attendance'
            filepath = os.path.join(output_dir, filename)

            # saving face image with timestamp 
            # if name not in saved_names:
            #     cv.imwrite(filepath, face)
            #     saved_names.add(name)

            facenet_results.append(facenet_result)
    print('section aaaaaaaaaaaaa')
    # Annotate frame with the bounding boxes and labels of detections within ROI
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    print('section bbbbbbbb')
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, text_from_facenet = facenet_results,labels=labels)

    # cv.imshow('Live Stream', annotated_frame)
    return annotated_frame

sv.process_video(
    source_path="output.mp4",
    target_path="roioutput.mp4",
    callback=callback
)
