import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

image_data = np.load('images_data.npy')
labels = np.load('labels.npy')

image_data = image_data.reshape(image_data.shape[0], 28, 28, 1)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)

labelsTrain = np.argmax(labelsTrain, axis=1)
labelsTest = np.argmax(labelsTest, axis=1)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(imageTrain, labelsTrain, epochs=20, validation_data=(imageTest, labelsTest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(imageTrain, labelsTrain, verbose=2)

plt.show()