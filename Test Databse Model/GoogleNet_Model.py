import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

image_data = np.load('images_data.npy')
labels = np.load('labels.npy')

imageTrain, imageTest, labelsTrain, labelsTest = train_test_split(image_data, labels, test_size=0.2)

num_classes = 2


def googlenet(input_shape, num_classes):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((1, 1)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((1, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


googlenet_model = googlenet(input_shape=(28, 28, 1), num_classes=num_classes)

googlenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = googlenet_model.fit(imageTrain, labelsTrain, epochs=20, validation_data=(imageTest, labelsTest))

test_loss, test_acc = googlenet_model.evaluate(imageTest, labelsTest, verbose=2)
print("\nTest Accuracy:", test_acc)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

train_loss, train_acc = googlenet_model.evaluate(imageTrain, labelsTrain, verbose=2)

plt.show()