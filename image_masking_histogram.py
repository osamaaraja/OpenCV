import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Loading the original image
img = cv.imread('cats.jpg')
cv.imshow('Original', img)

# creating a blank image of all zeros
blank_img = np.zeros((img.shape[:2]), dtype=np.uint8)

# converting the original image to gray scale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', gray_img)

# creating a mask
circle_blank = cv.circle(blank_img.copy(), ((img.shape[1]//2), (img.shape[0]//2)-220), 200, 255, -1)
circle_mask = cv.bitwise_and(gray_img, gray_img, mask=circle_blank)
cv.imshow('Mask', circle_mask)

# histogram of the original gray scale image
hist_gray = cv.calcHist([gray_img], [0], None, [256], [0, 256])

# histogram of the masked original gray scale image
hist_gray_masked = cv.calcHist([gray_img], [0], circle_mask, [256], [0, 256])


histograms = [hist_gray, hist_gray_masked]
titles = ['Grayscale Histogram without masking', 'Grayscale Histogram with masking']

fig, axs = plt.subplots(1, 2, figsize=(20, 10))

for ax, hist, title in zip(axs, histograms, titles):
    ax.set_title(title)
    ax.set_xlabel("Bins")
    ax.set_ylabel("Number of Pixels")
    ax.plot(hist)

fig.tight_layout()
plt.show()

# Histogram for the colored image without using the gray version of the image
color = ("b", "g", "r")
plt.figure(figsize=(20,10))
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()


cv.waitKey(0)



