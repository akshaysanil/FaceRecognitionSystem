# Face Attendance System

Welcome to the Face Attendance System! This project leverages advanced computer vision and machine learning techniques to automate the process of marking attendance. The system detects, tracks, and recognizes faces using entry and exit cameras, ensuring accurate and efficient attendance tracking.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Contributing](#contributing)

## Introduction

The Face Attendance System is designed to streamline the attendance process by using facial recognition technology. When a person enters the office, the system detects and tracks their face, and upon successful recognition, marks their attendance with a timestamp. The system also logs the exit time as the person leaves.

## Features

- **Real-time Face Detection**: Utilizing YOLOv8 for fast and accurate face detection.
- **Face Recognition**: Employing FaceNet for reliable face recognition.
- **Tracking**: Implementing ByteTracker for effective face tracking.
- **Machine Learning Algorithm**: Using SVM for classifying and recognizing faces.
- **Timestamp Logging**: Automatically logs entry and exit times.

## Technologies Used

- **Detection**: [YOLOv8](https://github.com/ultralytics/yolov8)
- **Recognition**: [FaceNet](https://github.com/davidsandberg/facenet)
- **Tracking**: [ByteTracker](https://github.com/ifzhang/ByteTrack)
- **Machine Learning Algorithm**: SVM (Support Vector Machine)
- **Supervision**: Supervision framework
- **Python , OpenCv,imageo**


## Installation

To get started with the Face Attendance System, follow these steps:

1. **Clone the repository**:
    ```sh
    [git clone https://github.com/akshaysanil/FaceRecognitionSystem.git]
    cd face-attendance-system
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download pre-trained models**:
   - YOLOv8-face weights: [Download here]([https://github.com/ultralytics/yolov8/releases](https://github.com/akanametov/yolo-face.git))
   
4. **Setup your configuration**:
    - Ensure that the paths to your models and any other configurations are correctly set in the configuration file.

## Usage

1. **Register a new person**:
   - Create newdataset and use
        - Dataset/emoplyee1/image1.jpg,image2.jpg
        - Dataset/emoplyee2/image1.jpg,image2.jpg


3. **Run the system**:
    ```sh
    python main.py
    ```
4. **Monitor the entry and exit cameras**:
   - The system will automatically detect, track, and recognize faces, marking attendance in the process.
   - All the marked attandence face-crop will save inside the *current_date* folder(it will create everyday) with the current timestamp.
   - Also every details will save inside *current_date.csv* with *name,unique_id,timestamp,hyperlink* for **Admin**.

## Demo


![image](https://github.com/akshaysanil/FaceRecognitionSystem/assets/104578088/781c97b0-a554-4c4f-84cd-0bb60097b1bf)
![image](https://github.com/akshaysanil/FaceRecognitionSystem/assets/104578088/fd87c70c-6337-44b9-873b-0ab1d8288b45)

### Demo video
Check out the demo video to see the Face Attendance System in action:
![](https://github.com/akshaysanil/FaceRecognitionSystem/blob/master/demo.jpg)

## Contributions 

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

