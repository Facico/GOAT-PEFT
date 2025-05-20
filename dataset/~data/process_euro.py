src_dir = f'euro_sat/2750'    # replace with the path to your dataset
dst_dir = f'./EuroSAT_splits'  # replace with the path to the output directory

import os
import shutil
import random

def create_directory_structure(dst_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(dst_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

def split_dataset(dst_dir, src_dir, classes, val_size=270, test_size=270):
    for cls in classes:
        class_path = os.path.join(src_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]
        
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'train', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'val', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dst_dir, 'test', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            # break

classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
create_directory_structure(dst_dir, classes)
split_dataset(dst_dir, src_dir, classes)