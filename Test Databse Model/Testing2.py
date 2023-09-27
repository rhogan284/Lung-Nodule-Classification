import numpy as np

images = np.load('labels.npy')

print(images.shape)

total_count = 0

unique_values, counts = np.unique(images, return_counts=True)

for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")
    total_count += count

print(total_count)