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

image_data = np.load('nodule_images.npy')
labels = np.load('malignancy_scores.npy')
print(image_data.shape)
print(labels.shape)

image_data = image_data.reshape(image_data.shape[0], 52, 52, 1)
labels = labels - 1

labels = tf.keras.utils.to_categorical(labels, 3)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)

def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True):
    shortcut = x

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x


input_tensor = Input(shape=(52, 52, 1))

x = layers.Conv2D(64, (7, 7), strides=2, padding="same")(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

x = residual_block(x, 64, conv_shortcut=False)
x = residual_block(x, 128, stride=2, conv_shortcut=True)
x = residual_block(x, 256, stride=2, conv_shortcut=True)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(3, activation="softmax")(x)

model = models.Model(inputs=input_tensor, outputs=x)

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(imageTrain, labelsTrain, epochs=150, validation_data=(imageTest, labelsTest))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()