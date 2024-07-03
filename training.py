import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mtcnn.mtcnn import MTCNN
detector = MTCNN()


# automate the preproseccing
class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()


    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')


faceloading = FACELOADING('facenet_files/dataset2')
X,Y = faceloading.load_classes()


# plt.figure(figsize = (32,24))
# for num,image in enumerate(X):
#   ncols = 4
#   nrows = len(Y)//ncols +1
#   plt.subplot(nrows,ncols,num+1)
#   plt.imshow(image)
#   plt.axis('off')

from keras_facenet import FaceNet
embedder = FaceNet()

def get_embedding(face_img):
  face_img = face_img.astype('float32') #3d (160,160)
  face_img = np.expand_dims(face_img,axis= 0)
  #4D (Nonex160,160,3)
  yhat = embedder.embeddings(face_img)
  return yhat[0] # 512d image (1x1x512)

EMBEDDED_X = []

for img in X:
  EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)


np.savez_compressed('faces_embeddings_done_for_officeMysr.npz',EMBEDDED_X,Y)

#svm model
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

plt.plot(EMBEDDED_X[0])
plt.ylabel(Y[0])

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)


from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_train, ypreds_train)
print('accuracy_score(Y_train, ypreds_train) :',accuracy_score(Y_train, ypreds_train))

accuracy_score(Y_test,ypreds_test)
print('accuracy_score(Y_test,ypreds_test) : ',accuracy_score(Y_test,ypreds_test))


# --------------------  testing an unknow image -----------------


# t_im = cv.imread("/content/drive/MyDrive/machine_test_mysur/face_recognition/ktpyUnknow.jpeg")
# t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
# x,y,w,h = detector.detect_faces(t_im)[0]['box']

# t_im = t_im[y:y+h, x:x+w]
# t_im = cv.resize(t_im, (160,160))
# test_im = get_embedding(t_im)

# test_im = [test_im]
# ypreds = model.predict(test_im)

# encoder.inverse_transform(ypreds)


import pickle
#save the model
with open('svm_model_160x160_Office_mysr.pkl','wb') as f:
    pickle.dump(model,f)