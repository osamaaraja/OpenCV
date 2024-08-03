import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ["Hugh Jackman", "Brad Pitt", "Eva Green", "Alessandra Daddario"]

# Load the trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load a test image
img = cv.imread(r'Faces_Val/Hugh Jackman/15.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]  # Corrected dimensions
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence {confidence}')

    cv.putText(img, str(people[label]), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()
