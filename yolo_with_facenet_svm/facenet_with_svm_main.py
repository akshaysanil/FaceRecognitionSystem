# Install necessary packages
# pip install mtcnn keras-facenet scikit-learn opencv-python

import cv2 as cv
import os
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MTCNN detector
detector = MTCNN()

# Initialize FaceNet embedder
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

# Load the precomputed embeddings and labels
data = np.load('faces_embeddings_done_4classes.npz')
EMBEDDED_X = data['arr_0']
Y = data['arr_1']

# If you need to re-encode the labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Load saved model
with open('svm_model_160x160.pkl', 'rb') as f:
    model = pickle.load(f)

def process_image(image_path):
    try:
        t_im = cv.imread(image_path)
        t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
        x, y, w, h = detector.detect_faces(t_im)[0]['box']
        t_im = t_im[y:y+h, x:x+w]
        t_im = cv.resize(t_im, (160, 160))
        test_im = get_embedding(t_im)
        
        return test_im
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_directory(directory_path):
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            embedding = process_image(image_path)
            if embedding is not None:
                embedding = np.expand_dims(embedding, axis=0)
                ypreds = model.predict(embedding)
                result = encoder.inverse_transform(ypreds)[0]
                results.append((filename, result))
    return results


def predict_image(image_path):
    embedding = process_image(image_path)
    if embedding is not None:
        embedding = np.expand_dims(embedding, axis=0)
        ypreds = model.predict(embedding)
        return encoder.inverse_transform(ypreds)
    return None

# Example usage
image_path = '/home/akshay/work/mysur_tests/machineTest1/facenet_new/test_images/Ashlesha P D272.jpg'
result = predict_image(image_path)
print(result)
