import os
import cv2 as cv
import numpy as np

people = ["Hugh Jackman", "Brad Pitt", "Eva Green", "Alessandra Daddario"]

DIR = r'Faces_Train'

# Verify if the directory paths are correct
print(f"Training directory contains: {os.listdir(DIR)}")

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print(f"Length of the features = {len(features)}")
print(f"Length of the labels = {len(labels)}")
print("Training Complete-------------------------")

# Convert to numpy arrays and save
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer and save the model
face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')

# Save features and labels arrays
np.save('features.npy', features)
np.save('labels.npy', labels)
