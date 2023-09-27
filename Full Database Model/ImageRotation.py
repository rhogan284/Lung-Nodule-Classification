import numpy as np
from scipy.ndimage import rotate

images = np.load('nodule_images.npy')
labels = np.load('malignancy_scores.npy')

selected_images = images[labels == 3]

rotated_images = np.array([rotate(image, angle=90, reshape=False) for image in selected_images])

augmented_images = np.vstack((images, rotated_images))

augmented_labels = np.hstack((labels, np.full(rotated_images.shape[0], 3)))

print(images.shape)
print(labels.shape)
print(augmented_images.shape)
print(augmented_labels.shape)

unique_values, counts = np.unique(augmented_labels, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")

indices_with_label_2 = np.where(augmented_labels == 2)[0]

random_indices_to_remove = np.random.choice(indices_with_label_2, size=460, replace=False)

augmented_images = np.delete(augmented_images, random_indices_to_remove, axis=0)
augmented_labels = np.delete(augmented_labels, random_indices_to_remove, axis=0)

print(augmented_images.shape)
print(augmented_labels.shape)

unique_values, counts = np.unique(augmented_labels, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")

all_rotated_images = np.array([rotate(image, angle=180, reshape=False) for image in augmented_images])

augmented_images2 = np.vstack((augmented_images, all_rotated_images))

augmented_labels2 = np.hstack((augmented_labels, augmented_labels))

print(augmented_images2.shape)
print(augmented_labels2.shape)

unique_values, counts = np.unique(augmented_labels2, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")

nodule_images_equal = np.array(augmented_images2)
malignancy_scores_equal = np.array(augmented_labels2)
np.save('nodule_images_equal_2', nodule_images_equal)
np.save('malignancy_scores_equal_2', malignancy_scores_equal)