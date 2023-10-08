import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import matplotlib.pyplot as plt
from skimage.transform import resize

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


all_scans = pl.query(pl.Scan).all()

nodule_images = []
malignancy_scores = []

for scan in all_scans:
    try:
        vol = scan.to_volume()
        nods = scan.cluster_annotations()
        nod_count = 0

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

            if cropped_vol_3d.shape[2] > 0:
                k = cropped_vol_3d.shape[2] // 2
                central_slice = cropped_vol_3d[:, :, k]

                # Resize the cropped image to 26x26 pixels
                resized_cropped = resize(central_slice, (26, 26))
                nodule_images.append(resized_cropped)
                malignancy = get_most_frequent_malignancy(anns)
                malignancy_scores.append(malignancy)
                nod_count += 1
                print(f"{scan}, nodule {nod_count} with malignancy {malignancy} added.")
            else:
                print("Skipped nodule due to empty cropped volume along z-axis.")

    except RuntimeError as e:
        print(f"Skipped scan for patient {scan.patient_id} due to error: {e}")
        continue

nodule_images_np = np.array(nodule_images)
malignancy_scores_np = np.array(malignancy_scores)
np.save('nodule_images_updated', nodule_images_np)
np.save('malignancy_scores_updated', malignancy_scores_np)
