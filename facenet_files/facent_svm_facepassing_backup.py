import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MTCNN detector
detector = MTCNN()

# Initialize FaceNet embedder
embedder = FaceNet()

# Load the saved embeddings and labels
data = np.load('faces_embeddings_done_4classes.npz')
EMBEDDED_X = data['arr_0']
Y = data['arr_1']

# Re-encode the labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Load saved SVM model
with open('svm_model_160x160.pkl', 'rb') as f:
    model = pickle.load(f)

def predict_face(image_path):
    try:
        # Read the image
        t_im = cv.imread(image_path)
        t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
        
        # # Detect face
        # x, y, w, h = detector.detect_faces(t_im)[0]['box']
        # face_img = t_im[y:y+h, x:x+w]
        
        # Resize and process the face image
        face_img = cv.resize(t_im, (160, 160))
        face_embedding = get_embedding(face_img)
        
        # Predict using the SVM model
        embedding = np.expand_dims(face_embedding, axis=0)
        ypreds = model.predict(embedding)
        
        # Inverse transform to get the predicted label
        result = encoder.inverse_transform(ypreds)[0]
        
        return result
    
    except Exception as e:
        print(f"Error predicting face: {e}")
        return None

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]


print(predict_face('face.png'))