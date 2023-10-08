import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from pylidc.utils import consensus


def get_most_frequent_malignancy(anns):
    class_mapping = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
    classes = [class_mapping[ann.malignancy] for ann in anns]

    most_frequent_class = max(set(classes), key=classes.count)
    class_counts = {cls: classes.count(cls) for cls in set(classes)}
    sorted_classes = sorted(class_counts, key=class_counts.get, reverse=True)

    if len(sorted_classes) > 1 and class_counts[sorted_classes[0]] == class_counts[sorted_classes[1]]:
        return 2
    else:
        return most_frequent_class


patient_id = 'LIDC-IDRI-0005'
scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()

nodule_images = []
malignancy_scores = []

for scan in scans:
    vol = scan.to_volume()
    nods = scan.cluster_annotations()

    for anns in nods:
        # Compute consensus mask with a 50% agreement level
        cmask, cbbox, _ = consensus(anns, clevel=0.5)

        # Define padding value
        padding = 0  # Increase the padding value

        # Define the bounding box using the consensus mask with added padding
        min_i, max_i = max(0, cbbox[0].start - padding), min(vol.shape[0], cbbox[0].stop + padding)
        min_j, max_j = max(0, cbbox[1].start - padding), min(vol.shape[1], cbbox[1].stop + padding)

        # Crop the 3D volume based on the adjusted bounding box
        cropped_vol_3d = vol[min_i:max_i, min_j:max_j, cbbox[2].start:cbbox[2].stop]

        # Extract the central slice from the cropped 3D volume
        k = cropped_vol_3d.shape[2] // 2
        central_slice = cropped_vol_3d[:, :, k]

        # Resize the cropped image to 26x26 pixels
        resized_cropped = resize(central_slice, (26, 26))
        nodule_images.append(resized_cropped)

        malignancy = get_most_frequent_malignancy(anns)
        malignancy_scores.append(malignancy)

nodule_images_np = np.array(nodule_images)
malignancy_scores_np = np.array(malignancy_scores)

num_nodules = len(nodule_images)

num_cols = num_nodules
num_rows = 1

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 5))

if num_nodules == 1:
    axes.imshow(nodule_images[0], cmap='gray')
    axes.set_title(f"Nodule 1\nMalignancy: {malignancy_scores[0]}")
    axes.axis('off')
else:
    for idx, (img, ax) in enumerate(zip(nodule_images, axes)):
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Nodule {idx+1}\nMalignancy: {malignancy_scores[idx]}")
        ax.axis('off')

plt.tight_layout()
plt.show()
print(nodule_images_np)