import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import *
import cv2 as cv
from datetime import datetime
import os
import uuid
import csv

model = YOLO("yolo_models/yolov8n-face.pt")  # updated version
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = []

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
    
    # Drawing the line
    height, width, _ = frame.shape  # The line will be in the middle of the frame
    line_y = height // 2
    start_point = (0, line_y)
    end_point = (width, line_y)
    color = (0, 255, 0)  # Green color in BGR
    thickness = 1  # Line thickness
    cv.line(annotated_frame, start_point, end_point, color, thickness)

    # Prepare CSV file
    current_date = datetime.now().strftime('%Y_%m_%d')
    output_dir = os.path.join('marked_attendance', current_date)
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f'{current_date}.csv')
    csv_header = ['UniqueID', 'Timestamp', 'Hyperlink']

    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    for detection in detections.xyxy:
        x1, y1, x2, y2 = map(int, detection[:4])
        face = frame[y1:y2, x1:x2]
        print('Getting face')

        facenet_result, result_probabilty = predict_face(face)
        name = facenet_result
        print('Name:', name)
        print('Result Probability:', result_probabilty)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        unique_id = str(uuid.uuid4())
        filename = f'{name}_{timestamp}.jpg'
        filepath = os.path.join(output_dir, filename)

        # Saving the attendance to the database (locally) when the person crosses the line
        if (y1 <= line_y <= y2) and name != 'Unknown' and name not in saved_names:
            cv.imwrite(filepath, face)
            first_name = filename.split('_')[0]  # Taking the first name from the filename
            print('First Name:', first_name)
            saved_names.append(first_name)

            # Writing to CSV
            hyperlink = os.path.abspath(filepath)
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([unique_id, timestamp, hyperlink])

        facenet_results.append(facenet_result)
        print('Facenet Results:', facenet_result)
        result_probabiltys.append(result_probabilty)
        print('Result Probabilities:', result_probabiltys)

    cv.imwrite('live_frame.png', annotated_frame)
    return label_annotator.annotate(annotated_frame, detections=detections, labels=labels, text_from_facenet=facenet_results, result_probability=result_probabiltys)

sv.process_video(
    source_path="result_harshitha.mp4",
    target_path="result/result_harshitha.mp4",
    callback=callback
)
