import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

DIRECTORY = r'C:\Users\HP\Downloads\archive\dogsandcats\valid' 
CATEGORIES = ['cats','dogs']

from keras.models  import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



IMG_SIZE = 90

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
        data.append([img_arr, label])
        
len(data)

random.shuffle(data)

X = []
Y = []

for features, labels in data:
    X.append(features)
    Y.append(labels)

X = np.array(X)
Y = np.array(Y)


X=X/255


model= Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(200, input_shape = X.shape[1:], activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


model.compile(optimizer = 'adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])


history=model.fit(X, Y, epochs=15, validation_split=0.1)


model.evaluate(X,Y)


plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
