


import cv2 as cv
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


embedder = FaceNet()

# Load the saved embeddings and labels
data = np.load('facenet_models/faces_embeddings_done_5members_V2.npz')
EMBEDDED_X = data['arr_0']
Y = data['arr_1']

# Re-encode the labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Load saved SVM model
with open('/home/akshay/work/mysur_tests/machineTest1/face_recognition_system/svm_models/svm_model_5memmbers_V2.pkl', 'rb') as f:
    model = pickle.load(f)
    print('model newclassified >>>>>>>>>>>>>>> : ',model)

with open('facenet/model/new_classifier_jun13.pkl', 'rb') as f:
    model = pickle.load(f)
    print('model newclassified >>>>>>>>>>>>>>> : ',model)
    


def predict_face(face_image):
    # Read the image
    cv.imwrite('face_from_yolo.png',face_image)
    t_im = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
    
    # Resize and process the face image
    face_img = cv.resize(t_im, (160, 160))
    face_embedding = get_embedding(face_img)
    
    # Predict using the SVM model
    embedding = np.expand_dims(face_embedding, axis=0)
    ypreds = model[0].predict(embedding)
    print('ypreds ---------------- : ',ypreds) # here we only getting the index
    ypreds_probabilty_list = model[0].predict_proba(embedding)
   

   #taking index and higest prob
    probabilities = ypreds_probabilty_list[0]
    print('probabilities,-------------',probabilities)
    
    highest_probability = max(probabilities)
    print('higest probabilty ---------- : ',highest_probability)

    # print('higest probabilty index ---------- : ',np.argmax(probabilities))
    # highest_probability_index = np.argmax(probabilities)

    
    #assinging result probability
    result_probability = highest_probability
    
    # Inverse transform to get the predicted label
    result = encoder.inverse_transform(ypreds)[0]
    print('result model.ypred ------------ :',result)
    
    # result = encoder.inverse_transform([highest_probability_index])
    # print(' encoder.inverse_transform(highest_probability_index) -------------:   ', encoder.inverse_transform([highest_probability_index]))
    # print('result model.ypred_proba ------------ :',result)

    # result = encoder.inverse_transform(ypreds)
    print('result ......................... : ',result)

    return result,result_probability

def get_embedding(face_img):
    face_img = face_img.astype('float32')
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)

    return yhat[0]

