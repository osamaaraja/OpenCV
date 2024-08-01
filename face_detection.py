import cv2 as cv

img = cv.imread('Sample_images/multiple_people.jpg')
#cv.imshow('person', img)

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('gray person', gray_img)

# creating a haarcascade variable
haar_cascade = cv.CascadeClassifier('haar_face.xml') # this xml is obtained from opencv repo on github

faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=7) # face rectangular coordinates

print(f"number of faces detected: {len(faces_rect)}")

# looping over the coordinates to form a rectangle around the face
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0), thickness=3)

cv.imshow('Detected faces',img)

cv.waitKey(0)