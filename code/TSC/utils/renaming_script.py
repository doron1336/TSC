import os.path
import pickle
from typing import Any

import numpy as np

from TSC.models.minirocket import fit, transform

baseDir = "UCRArchive_2018"
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), baseDir, item) for item in
            os.listdir(os.path.join(os.getcwd(), baseDir))]


def rename_files(directory, old_string, new_string):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if old_string in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(old_string, new_string)
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


for dataset_path in datasets:
    print(dataset_path)
    dataset_name = dataset_path.split("/")[-1]
    if dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted":
        continue
    rename_files(dataset_path, old_string="handmovement", new_string=dataset_name)


