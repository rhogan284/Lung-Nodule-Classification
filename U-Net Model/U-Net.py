import pylidc as pl
import numpy as np
import cv2

# Load the first 100 annotations
annotations = pl.query(pl.Annotation).limit(500).all()

# Extract CT slices and masks for the annotations
volumes = []
masks = []

padding = [(30, 10), (10, 25), (0, 0)]

# Define a target shape for the images
TARGET_SHAPE = (128, 128)

for ann in annotations:
    vol = ann.scan.to_volume()
    mask = ann.boolean_mask(pad=padding)
    bbox = ann.bbox(pad=padding)

    # Check the shape of the volume slice
    slice_shape = vol[bbox].shape

    # Extract the appropriate slice
    if slice_shape[2] >= 3:
        slice_vol = vol[bbox][:, :, slice_shape[2] // 2]
        slice_mask = mask[:, :, slice_shape[2] // 2]
    else:
        slice_vol = vol[bbox][:, :, -1]
        slice_mask = mask[:, :, -1]

    # Convert boolean mask to uint8
    slice_mask = (slice_mask * 255).astype(np.uint8)

    # Resize the slices to the target shape
    resized_vol = cv2.resize(slice_vol, TARGET_SHAPE)
    resized_mask = cv2.resize(slice_mask, TARGET_SHAPE)

    volumes.append(resized_vol)
    masks.append(resized_mask)

# Convert lists to numpy arrays
volumes = np.array(volumes)
masks = np.array(masks)
np.save("volumes", volumes)
np.save("masks", masks)

print(volumes.shape, masks.shape)
