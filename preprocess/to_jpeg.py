"""
Usage:
# Create train data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/train -o [PATH_TO_ANNOTATIONS_FOLDER]/train_labels.csv

# Create test data:
python xml_to_csv.py -i [PATH_TO_IMAGES_FOLDER]/test -o [PATH_TO_ANNOTATIONS_FOLDER]/test_labels.csv
"""

import os
import glob
import pandas as pd
import argparse
import numpy as np
from PIL import Image


def to_jpeg(path):
    """Iterates through each image in folder and convert them to .jpeg format

    Parameters:
    ----------
    path : {str}
        Path to images folder
    """

    images = os.listdir(path)
    for image in images:
        if image.split(".")[-1] == "xml":
            continue
        img_path = os.path.join(path, image)
        img = np.array(Image.open(img_path))
        os.remove(img_path)
        if img.shape[-1] < 3:
            continue
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        new_name = ".".join(img_path.split(".")[:-1]) + ".jpeg"
        Image.fromarray(img).save(new_name)


def main():
    parser = argparse.ArgumentParser(description="jpeg converter")
    parser.add_argument("-p", "--folderPath", help="Path to the image folder", type=str)
    args = parser.parse_args()

    assert os.path.isdir(args.folderPath)
    to_jpeg(args.folderPath)


if __name__ == "__main__":
    main()
