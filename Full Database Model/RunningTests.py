import numpy as np
import pandas as pd

images = np.load('nodule_images.npy')
labels = np.load('malignancy_scores.npy')


unique_values, counts = np.unique(labels, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")

