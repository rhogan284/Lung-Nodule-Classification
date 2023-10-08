import numpy as np

images = np.load('nodule_images_updated_3.npy')
labels = np.load('malignancy_scores_updated_3.npy')

unique_values, counts = np.unique(labels, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")