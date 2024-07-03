import numpy as np
import supervision as sv
import torch, cv2, os
from datetime import datetime
from facenet_with_webcam import *
 


model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/face_rec.pt')
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

saved_names = set()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = sv.Detections.from_yolov5(results)
    # print(len(detections))
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
        
    except Exception as E:
        print(E) 
    
    # Draw a line on the frame
    start_point = (0, 600)  # Example start point (x, y)
    end_point = (490, 600)  # Example end point (x, y)
    color = (0, 255, 0)  # Line color in BGR (green in this case)
    thickness = 2  # Line thickness
    annotated_frame = cv2.line(annotated_frame, start_point, end_point, color, thickness)

    # Check if any detection touches the line and save the face
    for detection,label in zip(detections.xyxy,labels):
        print(label)
        x1, y1, x2, y2 = map(int, detection[:4])

        name = label.split(' ')[1]
        print('name : ',name)
        if name not in saved_names and y1 <= 600 <= y2:
            print('saved_names ',saved_names)
            # print('condition trueeeeeeeeeeeeeeeeeeeee................')
            face = frame[y1:y2, x1:x2]
            # timestamp = datetime.now().strftime("%Y_%m_%d")
            timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

            print('label full : ',label)
            print('label with [1]',label.split(' ')[1])
            filename = f"{name}_{timestamp}_{x1}_{y1}.jpg"
            print('filename    : ',filename)
            output_dir = '/home/akshay/work/machineTest1/dumpedimages2'
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, face)
            saved_names.add(name)

    return annotated_frame



sv.process_video(
    source_path="/home/akshay/work/machineTest1/videos1_1/merged.mp4",
    target_path="./result3.mp4",
    callback=callback
)



