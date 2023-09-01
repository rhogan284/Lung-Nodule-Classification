import os
import pydicom
from PIL import Image
import numpy as np


def convert_dcm_to_jpg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    for file in files:
        if file.endswith('.dcm'):
            dcm_image_path = os.path.join(input_folder, file)
            dicom_data = pydicom.dcmread(dcm_image_path)

            if 'PixelData' in dicom_data:
                array = dicom_data.pixel_array

                if array.dtype != np.uint8:
                    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

                image = Image.fromarray(array)

                jpg_image_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.jpg')
                image.save(jpg_image_path)
                print(f"Saved: {jpg_image_path}")
            else:
                print(f"No PixelData found in {file}. Skipping.")


if __name__ == "__main__":
    INPUT_FOLDER = 'Test CT Scans/3000562.000000-NA-07402'
    OUTPUT_FOLDER = 'Test CT Scans/Adjusted Images'

    convert_dcm_to_jpg(INPUT_FOLDER, OUTPUT_FOLDER)
