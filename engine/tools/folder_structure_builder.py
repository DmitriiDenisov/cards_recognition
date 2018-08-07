import glob
import shutil
import random
import os
import numpy as np

DATA_PATH = r'\\rumos0104\Data2\DAI\Projects\CardsMobile\data'
BARCODE_TYPE = 'CODE_128'
classes_folders = glob.glob(os.path.join(DATA_PATH, BARCODE_TYPE, 'train_source', '*'))

for idx, folder in enumerate(classes_folders):
    class_name = os.path.basename(folder)
    print(idx, class_name)

    train_target_folder = os.path.join(DATA_PATH, BARCODE_TYPE, 'train', class_name)
    test_target_folder = os.path.join(DATA_PATH, BARCODE_TYPE, 'val', class_name)
    os.mkdir(train_target_folder)
    os.mkdir(test_target_folder)

    file_list = np.array(glob.glob(os.path.join(folder, '*')))

    train_indexes = set(random.sample(range(len(file_list)), len(file_list)*2//3))
    test_indexes = set(range(len(file_list))) - train_indexes

    for filename in file_list[list(train_indexes)]:
        shutil.copy(filename, train_target_folder)
    for filename in file_list[list(test_indexes)]:
        shutil.copy(filename, test_target_folder)