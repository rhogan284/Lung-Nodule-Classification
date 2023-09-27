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

def hu_to_grayscale(image, window_center, window_width):
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2

    # Clip the image to the window
    image = np.clip(image, min_val, max_val)

    # Normalize to 0-255
    image_normalized = ((image - min_val) / (max_val - min_val)) * 255.0

    # Convert to 8-bit unsigned integer
    grayscale_image = np.uint8(image_normalized)
    return grayscale_image


all_scans = pl.query(pl.Scan).all()

nodule_images = []
malignancy_scores = []

for scan in all_scans:
    try:
        vol = scan.to_volume()
        nods = scan.cluster_annotations()
        nod_count = 0

        for anns in nods:
            cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])
            cropped_vol = vol[cbbox]

            if cropped_vol.shape[2] > 0:
                cropped_img_centered = crop_or_pad_to_center(cropped_vol[:, :, cropped_vol.shape[2] // 2], 52)
                hu_image = np.array(cropped_img_centered)
                window_center = -550
                window_width = 1550
                grayscale_image = hu_to_grayscale(hu_image, window_center, window_width)
                nodule_images.append(grayscale_image)
                malignancy = get_most_frequent_malignancy(anns)
                malignancy_scores.append(malignancy)
                nod_count += 1
                print(f"{scan}, nodule {nod_count} with malignancy {malignancy} added.")
            else:
                print(f"Skipped nodule {nod_count} for patient {scan.patient_id} due to empty cropped volume.")
    except RuntimeError as e:
        print(f"Skipped scan for patient {scan.patient_id} due to error: {e}")
        continue

nodule_images_np = np.array(nodule_images)
malignancy_scores_np = np.array(malignancy_scores)
np.save('nodule_images', nodule_images_np)
np.save('malignancy_scores', malignancy_scores_np)
