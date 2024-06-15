import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import *
import cv2 
from datetime import datetime
import os
from imutils.video import VideoStream
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO model
model = YOLO("/home/akshay/Downloads/yolov8n-face.pt")

# Initialize tracker and annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
saved_names = set()

# Define callback function for processing each frame
def callback(frame: np.ndarray, frame_id: int) -> np.ndarray:
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

    for detection, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = map(int, detection[:4])
        face = frame[y1:y2, x1:x2]
        logging.info('Getting face for recognition')
        facenet_result = predict_face(face)
        name = facenet_result
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename = f'{name}_{timestamp}.jpg'
        logging.info(f'Filename: {filename}')
        # output_dir = '/home/prixgen-gpu/Music/Face_reco_from_akshay/facenet_new_with_yolo/result'
        # filepath = os.path.join(output_dir, filename)

        # # Saving face image with timestamp
        # if name not in saved_names:
        #     cv2.imwrite(filepath, face)
        #     saved_names.add(name)

        facenet_results.append(facenet_result)

    # Annotate frame with FaceNet results
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels, text_from_facenet=facenet_results)

# Start video stream from IP camera
logging.info('Starting video stream...')
vidObj = VideoStream('rtsp://admin:Ashlesha123@192.168.0.170').start()
time.sleep(2.0)  # Allow camera to warm up

# Get frame size information
frame = vidObj.read()
frame_height, frame_width = frame.shape[:2]

# Initialize video writer
output_path = 'result3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = 20  # Frames per second
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not vidObj.stream.isOpened():
    logging.error('Unable to open video stream: rtsp://admin:Ashlesha123@192.168.0.170')
else:
    logging.info('Successfully opened video stream')

    while True:
        # Capture frame-by-frame
        frame = vidObj.read()
        if frame is None:
            logging.error('Failed to capture frame')
            break

        # Process the frame
        processed_frame = callback(frame, frame_id=0)

        # Write the frame to the video file
        out.write(processed_frame)

        # Display the resulting frame
        cv2.imshow('Live Stream', processed_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video stream, video writer, and close windows
vidObj.stop()
out.release()
cv2.destroyAllWindows()
logging.info('Video stream stopped and windows closed')
