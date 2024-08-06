import os
import caer
import canaro
import keras.src.utils
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt


# All the images in the data have to be resized to have a particular size
IMG_SIZE = (80, 80)

# Setting the number of channels of the image to 1, making it grayscale
channels = 1

# Setting the path to the base directory
char_path = r"Simpsons_kaggle/archive/simpsons_dataset"

# Grabbing the images data of each character and storing it in a dictionary
char_dict = {}
for char in os.listdir(char_path):
    char_full_path = os.path.join(char_path, char)
    if os.path.isdir(char_full_path):  # Ensure it's a directory
        char_dict[char] = len(os.listdir(char_full_path))

# Sorting the dictionary in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
print(char_dict)

# grabbing the names of the first ten elements in the dictionary and storing them in a list
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count +=1
    if count >=10:
        break

print(characters)

# creating the training data
# the labels are according to the indexes
train = caer.preprocess_from_dir(char_path, characters, channels=channels,IMG_SIZE=IMG_SIZE, isShuffle=True)

# separating the training set into feature set and labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# normalizing the features in the range of (0,1)
featureSet = caer.normalize(featureSet)

# one hot encoding the labels
labels = keras.utils.to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)

del train
del featureSet
del labels
gc.collect()

# image data generator
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=32)

# creating the model

#######################################################################################################################
# NOTE: Edit the createSimpsonsModel by changing the optimizer to
# optimizer = SGD(learning_rate=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
#######################################################################################################################
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),
                                          loss='binary_crossentropy', decay=1e-6, learning_rate=0.001,
                                          momentum=0.9, nesterov=True)
print(model.summary())


training = model.fit(train_gen, steps_per_epoch=len(x_train)//32, epochs=10,
                     validation_data=(x_val, y_val),
                     validation_steps=len(y_val)//32)

# using openCV to test now how good the model is
test_path = r'Simpsons_kaggle/archive/kaggle_simpson_testset/kaggle_simpson_testset/bart_simpson_6.jpg'
img = cv.imread(test_path)
plt.imshow(img)
plt.show()

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, channels)

    return img

prediction = model.predict(prepare(img))
print(characters[np.argmax(prediction[0])])


