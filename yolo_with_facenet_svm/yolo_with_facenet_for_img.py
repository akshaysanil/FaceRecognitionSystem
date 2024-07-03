import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from test_images.facent_svm_facepassing import predict_face

# Load the YOLO model
model = YOLO("/home/akshay/Downloads/yolov8n-face.pt")

# Initialize the tracker and annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def process_image(image_path: str, output_path: str) -> None:
    # Read the image
    frame = cv2.imread(image_path)
    
    # Ensure the image was loaded correctly
    if frame is None:
        print("Error: Image not loaded correctly.")
        return
    
    # Perform detection
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Update tracker with detections
    detections = tracker.update_with_detections(detections)
    
    # Prepare labels for annotations
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    
    # Annotate the frame with bounding boxes
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    
    # Iterate over detected faces and perform face recognition
    for detection in detections.xyxy:
        x1, y1, x2, y2 = map(int, detection[:4])
        face = frame[y1:y2, x1:x2]
        cv2.imwrite('face.png',face)

        print('Getting face...')
        result = predict_face(face)
        print(result)
    
    # Annotate the frame with labels
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels, text_from_facenet = result)
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated image saved to {output_path}")

# Process the image
process_image(
    image_path="testingImg.jpeg",
    output_path="result.png"
)
