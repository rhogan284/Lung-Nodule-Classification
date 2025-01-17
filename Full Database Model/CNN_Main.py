import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout

image_data = np.load('nodule_images_updated_2.npy')
labels = np.load('malignancy_scores_updated_2.npy')

image_data = image_data.reshape(image_data.shape[0], 26, 26, 1)
labels = labels - 1

labels = tf.keras.utils.to_categorical(labels, 3)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(26, 26, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(imageTrain, labelsTrain, epochs=100, validation_data=(imageTest, labelsTest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(imageTrain, labelsTrain, verbose=2)

plt.show()