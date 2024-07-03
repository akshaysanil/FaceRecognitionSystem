import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import *
import cv2 as cv
from datetime import datetime
import os

model = YOLO("yolo_models/yolov8n-face.pt")  # Updated version
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

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

    facenet_results = []  # Initialize a list to collect face recognition results
    result_probabiltys = []

    for detection, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = map(int, detection[:4])
        face = frame[y1:y2, x1:x2]
        print('Getting face')
        
        # facenet_result = predict_face(face)
        facenet_result, result_probabilty = predict_face(face)
        name = facenet_result
        
        print('Name: ', name)
        print('Result probability: ', result_probabilty)
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename = f'{name}_{timestamp}.jpg'
        print('Filename: ', filename)
        output_dir = 'marked_attendance'
        filepath = os.path.join(output_dir, filename)

        if name != 'Unknown' and name not in saved_names:
            cv.imwrite(filepath, face)
            first_name = filename.split('_')[0]  # Extract the first name
            print('First name: ', first_name)
            saved_names.add(first_name)

        facenet_results.append(facenet_result)
        print('Facenet results: ', facenet_results)
        result_probabiltys.append(result_probabilty)
        print('Result probabilities: ', result_probabiltys)

    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels,
        text_from_facenet=facenet_results, result_probability=result_probabiltys)

def process_rtsp_stream(rtsp_url: str):
    cap = cv.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        annotated_frame = callback(frame, 0)
        cv.imshow("RTSP Stream", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Replace 'your_rtsp_stream_url' with your actual RTSP stream URL
rtsp_stream_url = 'rtsp://admin:Ashlesha123@192.168.0.170'
process_rtsp_stream(rtsp_stream_url)
