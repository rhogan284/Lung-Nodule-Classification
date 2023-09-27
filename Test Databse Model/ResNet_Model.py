import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import schedules, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

image_data = np.load('images_data.npy')
labels = np.load('labels.npy')

image_data = image_data.reshape(image_data.shape[0], 28, 28, 1)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)

labelsTrain = np.argmax(labelsTrain, axis=1)
labelsTest = np.argmax(labelsTest, axis=1)

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    shortcut = x

    # If a convolution is needed for the shortcut (due to change in dimensions)
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    # First Convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second Convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adding the shortcut to the main path
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x


input_tensor = Input(shape=(28, 28, 1))

x = layers.Conv2D(64, (7, 7), strides=2, padding="same")(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

x = residual_block(x, 64, conv_shortcut=False)
x = residual_block(x, 128, stride=2, conv_shortcut=True)
x = residual_block(x, 256, stride=2, conv_shortcut=True)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=input_tensor, outputs=x)

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(imageTrain, labelsTrain, epochs=20, validation_data=(imageTest, labelsTest))

plt.plot(history.history['loss'], label='accuracy')
plt.plot(history.history['val_loss'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()
