import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle

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

def process_image(image):
    try:
        x, y, w, h = detector.detect_faces(image)[0]['box']
        face = image[y:y+h, x:x+w]
        face = cv.resize(face, (160, 160))
        embedding = get_embedding(face)
        return embedding
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Initialize webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(frame_rgb)
    
    for face in faces:
        x, y, w, h = face['box']
        face_rgb = frame_rgb[y:y+h, x:x+w]
        face_rgb = cv.resize(face_rgb, (160, 160))
        embedding = get_embedding(face_rgb)
        
        if embedding is not None:
            embedding = np.expand_dims(embedding, axis=0)
            ypreds = model.predict(embedding)
            identity = encoder.inverse_transform(ypreds)[0]
            
            # Draw bounding box and label on the original frame
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, identity, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the frame
    cv.imshow('Face Recognition', frame)
    
    # Exit on pressing 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv.destroyAllWindows()
