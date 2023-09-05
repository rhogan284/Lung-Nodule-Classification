import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import matplotlib.pyplot as plt


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

# noinspection PyTypeChecker
def crop_or_pad_to_center(img, target_size):
    y, x = img.shape
    cy, cx = y // 2, x // 2

    y1, y2 = max(0, cy - target_size // 2), min(y, cy + target_size // 2)
    x1, x2 = max(0, cx - target_size // 2), min(x, cx + target_size // 2)

    cropped = img[y1:y2, x1:x2]

    dy, dx = target_size - cropped.shape[0], target_size - cropped.shape[1]
    pad_y1, pad_y2 = (dy + 1) // 2, dy // 2
    pad_x1, pad_x2 = (dx + 1) // 2, dx // 2

    padded = np.pad(cropped, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant')
    return padded


patient_id = 'LIDC-IDRI-0012'

scans_for_first_patient = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()

nodule_images = []
malignancy_scores = []

for scan in scans_for_first_patient:
    vol = scan.to_volume()
    nods = scan.cluster_annotations()

    for anns in nods:
        cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])
        cropped_vol = vol[cbbox]

        cropped_img_centered = crop_or_pad_to_center(cropped_vol[:, :, cropped_vol.shape[2] // 2], 52)

        nodule_images.append(cropped_img_centered)
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