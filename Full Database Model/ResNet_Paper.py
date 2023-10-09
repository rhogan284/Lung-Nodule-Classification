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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

image_data = np.load('nodule_images_updated_2.npy')
labels = np.load('malignancy_scores_updated_2.npy')

image_data = image_data.reshape(image_data.shape[0], 26, 26, 1)
labels = labels - 1

labels = tf.keras.utils.to_categorical(labels, 3)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)
train_generator = datagen.flow(imageTrain, labelsTrain, batch_size=64)


# Define the Residual Block
def cross_residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, l2_reg=5e-5):
    shortcut = x
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(x)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

# Define the Cross-Level Fusion Layer
def cross_level_fusion(inputs):
    x1, x2, x3 = inputs

    # Pool x1 to bring its dimensions to 2x2
    x1_pooled = layers.MaxPooling2D(pool_size=(3, 3))(x1)

    # Pool x2 to bring its dimensions to 2x2
    x2_pooled = layers.MaxPooling2D(pool_size=(2, 2))(x2)

    # Concatenate the pooled x1, upsampled x2, and x3 along the channel axis
    concatenated = layers.Concatenate(axis=-1)([x1_pooled, x2_pooled, x3])

    # 1x1 conv to mix features
    fused = layers.Conv2D(x3.shape[-1], (1, 1), activation='relu')(concatenated)
    return fused


input_tensor = Input(shape=(26, 26, 1))

# Initial Convolution
x = layers.Conv2D(32, (7, 7), strides=2, padding="same", kernel_regularizer=l2(5e-5))(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

# Residual Blocks with multi-level feature extraction
x1 = cross_residual_block(x, 32, conv_shortcut=False)
x2 = cross_residual_block(x1, 64, stride=2, conv_shortcut=True)
x3 = cross_residual_block(x2, 128, stride=2, conv_shortcut=True)

# Cross-Level Fusion
fused = cross_level_fusion([x1, x2, x3])

# Final Layers
x = layers.GlobalAveragePooling2D()(fused)
x = layers.Dense(3, activation="softmax")(x)

# Construct the final model
cross_resnet_model = models.Model(inputs=input_tensor, outputs=x)

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

cross_resnet_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=['accuracy'])

batch_size = 64
steps_per_epoch = len(imageTrain) // batch_size
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = cross_resnet_model.fit(train_generator, epochs=150, validation_data=(imageTest, labelsTest),
                                 steps_per_epoch=steps_per_epoch, callbacks=[early_stopping])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()
