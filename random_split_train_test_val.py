import os
import shutil
import random

def random_split(source_folder, folder1, folder2, folder3, probabilities):
    # Laver liste over filer i den samlede benign-mappe
    files = os.listdir(source_folder)

    # Fordeler filer 
    for file in files:
        probability = random.uniform(0, 1)
        if probability <= probabilities[0]:
            shutil.copy(os.path.join(source_folder, file), os.path.join(folder1, file))
        elif probability <= probabilities[0] + probabilities[1]:
            shutil.copy(os.path.join(source_folder, file), os.path.join(folder2, file))
        else:
            shutil.copy(os.path.join(source_folder, file), os.path.join(folder3, file))

# Angiver mapperne
source_folder = 'kaggle_dataset\\collected\\benign'
folder1 = 'kaggle_dataset\\train\\benign'
folder2 = 'kaggle_dataset\\test\\benign'
folder3 = 'kaggle_dataset\\val\\benign'

# Andelen af hele datasættet vi vil have i hver mappe
probabilities = [0.8, 0.1, 0.1]

# bruger funktionen på mapperne
random_split(source_folder, folder1, folder2, folder3, probabilities)