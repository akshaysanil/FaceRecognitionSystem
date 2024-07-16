import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet.facent_svm_rec_passing import * 
import gradio as gr
import cv2 as cv
from datetime import datetime
import uuid
import csv
import os

model = YOLO("yolo_models/yolov8n-face.pt") # Updated version
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = []

def callback(frame: np.ndarray) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    
    try:
        annotated_frame = box_annotator.annotate(
            frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)
        print('inside the try >>>>>>')
    except Exception as E:
        print(E) 

    facenet_results = []  # Initialize a list to collect face recognition results
    result_probabilities = []

    # Drawing the line
    height, width, _ = frame.shape # the line will be in the middle of the frame
    line_y = (height // 2) - 50
    start_point = (0, line_y) 
    end_point = (width, line_y)
    color = (0, 255, 0)  # Green color in BGR
    thickness = 1  # Line thickness
    cv.line(annotated_frame, start_point, end_point, color, thickness)

    for detection, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = map(int, detection[:4])
        
        face = frame[y1:y2, x1:x2]
        print('getting face')
        
        facenet_result, result_probability = predict_face(face)
        name = facenet_result
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename = f'{name}_{timestamp}.jpg'

        current_date = datetime.now().strftime('%Y_%m_%d')
        output_dir = os.path.join('marked_attendance', current_date)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        hyperlink = os.path.abspath(filepath)

        # CSV
        csv_file_path = os.path.join(output_dir, f"{current_date}_attendance_sheet.csv")
        csv_header = ['Name', 'UniqueID', 'Timestamp', 'Hyperlink']
        unique_id = str(uuid.uuid4())
        print('unique_id >>>>>>>>>>>>>>>>>>> : ', unique_id)

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(csv_header)

        # Saving the attendance to the database (locally) when the person crosses the line
        if (y1 <= line_y <= y2) and result_probability >= 0.87 and name not in saved_names:
            cv.imwrite(filepath, face)
            first_name = filename.split('_')[0] # Extracting the first name from the filename
            print('first_name >>>>>>>>>>>>>>>>>>> : ', first_name)

            saved_names.append(first_name)

            # Writing to CSV
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([first_name, unique_id, timestamp, hyperlink])

        facenet_results.append(facenet_result)
        result_probabilities.append(result_probability)

    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels,
        text_from_facenet=facenet_results, result_probability=result_probabilities)

    cv.imwrite('live.png', annotated_frame)
    return annotated_frame

# Replace with your RTSP link
rtsp_link = "rtsp://your_rtsp_link"

# Capture video from RTSP link
cap = cv.VideoCapture(rtsp_link)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = callback(frame)
    cv.imshow('RTSP Feed', annotated_frame)

    # Press 'q' to exit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

