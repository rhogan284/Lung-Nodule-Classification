import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import schedules, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data = np.load('nodule_images_updated_3.npy')
labels = np.load('malignancy_scores_updated_3.npy')

image_data = image_data.reshape(image_data.shape[0], 26, 26, 1)
converted_array = np.where(labels == 1, 0, 1)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, converted_array, test_size=0.2)
train_generator = datagen.flow(imageTrain, labelsTrain, batch_size=32)

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

input_tensor = Input(shape=(26, 26, 1))

x = layers.Conv2D(64, (7, 7), strides=2, padding="same")(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

x = residual_block(x, 64, conv_shortcut=False)
x = residual_block(x, 128, stride=2, conv_shortcut=True)
x = residual_block(x, 256, stride=2, conv_shortcut=True)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=input_tensor, outputs=x)

lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9)
optimizer = SGD(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

steps_per_epoch = len(imageTrain) // 32
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_generator, epochs=100, validation_data=(imageTest, labelsTest),
                    steps_per_epoch=steps_per_epoch, callbacks=[early_stopping])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()
