import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

images = np.load('images_data.npy')
values = np.load('labels.npy')

fig, axes = plt.subplots(10, 10, figsize=(15, 12))

for i, ax in enumerate(axes.ravel()):  # .ravel() flattens the 4x5 axes array into a 1D array
    ax.imshow(images[i], cmap='gray')
    ax.set_title(f"Value: {values[i]}")
    ax.axis('off')  # Turn off axis numbers and ticks

plt.tight_layout()
plt.show()

print(images[0])