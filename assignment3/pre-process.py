import os
import shutil
from glob import glob
from PIL import Image

if __name__ == '__main__':
    source_path = 'MNIST_dataset/MNIST_variable_resolution/mnist-varres/**'
    target_path = 'MNIST_dataset/MNIST_variable_resolution/mnist-varres-pre-processed'

    for file in glob(source_path, recursive=True):
        if os.path.isdir(file):
            continue
        file_info = file.split('MNIST_dataset/MNIST_variable_resolution/mnist-varres/')[1]
        target_type, target_class, file_name = file_info.split('/')
        print(file_info)
        with Image.open(file) as img:
            width, height = img.size
            path_dir = f'{target_path}/{target_type}/{width}/{target_class}'
            os.makedirs(path_dir, exist_ok=True)
            target_file = f'{path_dir}/{file_name}'
            if os.path.exists(target_file):
                continue
            shutil.copy(file, target_file)
