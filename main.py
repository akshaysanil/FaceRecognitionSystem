

import numpy as np
import supervision as sv
from ultralytics import YOLO
from facenet_files.facent_svm_rec_passing import * 
from supervision.annotators import core

import gradio as gr

import cv2 as cv
from datetime import datetime

import uuid
import csv


model = YOLO("yolo_models/yolov8n-face.pt") #updated version
tracker = sv.ByteTrack()
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
# saved_names = set()
saved_names = []

trackeri_d = 1
def callback(frame: np.ndarray, _: int) -> np.ndarray:
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
        # print('inside the try >>>>>>')

    except Exception as E:
        print(E) 

    # annotated_frame = box_annotator.annotate(
    #     frame.copy(), detections=detections)
    
    facenet_results = []  # Initialize a list to collect face recognition results
    result_probabiltys = []
    
    # drawing the line
    height, width, _ = frame.shape # the line will be in the middle of the frame
    line_y = (height // 2) - 50
    start_point = (0, line_y) 
    end_point = (width, line_y)
    color = (0, 255, 0)  # Green color in BGR
    thickness = 1  # Line thickness
    cv.line(annotated_frame, start_point, end_point, color, thickness)

    for detection,label in zip(detections.xyxy,labels):
        x1, y1, x2, y2 = map(int, detection[:4])
        # print('labels  >>>>>> : ',label)
        
        # extracting the detected face
        face = frame[y1:y2, x1:x2]
        print('getting face ################')

        # facenet_result = predict_face(face)
        print('Passing Rxtracted face to Recogition model ################ ')
        facenet_result, result_probabilty = predict_face(face)
        print('Recognition complete ################')
        name = facenet_result
        
        timestamp = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
        filename = f'{name}_{timestamp}.jpg'

        current_date = datetime.now().strftime('%Y_%m_%d')
        output_dir = os.path.join('marked_attendance',current_date)
        os.makedirs(output_dir,exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        hyperlink = os.path.abspath(filepath)



        """ here I addinig new .csv file in each current date folder which contains - 
        >>> Empoloyee name, unique Id, Timstamp, and Hyperlink(the path to the face croped image).
        * It will help the admin to check all the people attendance in a single file. 
        * Also it will help when we are using database to reduce the space.
        """

        #csv file creation
        csv_file_path = os.path.join(output_dir,f"{current_date}_attendance_sheet.csv")
        csv__header = ['Name','UniqueID', 'Timestamp','Hyperlink']
        unique_id = str(uuid.uuid4())
        # print('unique_id >>>>>>>>>>>>>>>>>>> : ',unique_id)

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, mode='w',newline='') as file:
                writer = csv.writer(file)
                writer.writerow(csv__header)


# saving the attandance to the database(locally) when the person cross the line
        # if (y1 <= line_y <= y2) and name != 'Unknown' and name not in saved_names :
        if (y1 <= line_y <= y2) and result_probabilty >=0.87 and name not in saved_names :

            cv.imwrite(filepath,face)
            first_name = filename.split('_'[0]) # taking the firstname from 
            # print('first_name >>>>>>>>>>>>>>>>>>> : ',first_name)

            # first_name[0] means only the exact name without timestamp
            saved_names.append(first_name[0])

            #writing to csv
            with open(csv_file_path,mode='a',newline='') as file:
                writer = csv.writer(file)
                writer.writerow([first_name[0],unique_id,timestamp,hyperlink])




        # facenet_results.append(facenet_result)[0]
        facenet_results.append(facenet_result)
        # print('facenet_results   :  ',facenet_result)
        result_probabiltys.append(result_probabilty)
        # print('result_probabiltys  :  ',result_probabiltys)


    # print('labels >>>>>>>>>>>>>>> : ',labels)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels,
        text_from_facenet = facenet_results,result_probability = result_probabiltys)
    
    '''cv2.imshow is not woring in my env if you are using this ,try this
    >>> cv2.imshow('live frame',annotated_frame)       '''
    

    cv.imwrite('live.png',annotated_frame)
  
    return annotated_frame



sv.process_video(
    source_path="test_datas/testing_video.mp4",
    target_path="result_datas/testig_video_result.mp4",
    callback=callback
)

#testing gradio

 
# ------------------------------- core file edited in this path -------------------------------------------
# /home/akshay/anaconda3/envs/py38/lib/python3.8/site-packages/supervision/annotators/core.py
# rtsp://admin:Ashlesha123@192.168.0.170
# conda : py38new
# run python3 yolo_with_facenet_main.py