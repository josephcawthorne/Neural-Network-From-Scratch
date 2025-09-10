
# In this file, I am going to try and write a couple of helper functions for working with my dataset.
# I am going to try and read the Fashion-MNIST IDX files (images + labels).
# First of, I am going to open the files and then I will try to parse them in a very basic way.

import os
import struct

# I want to keep this pretty simple. I will try to just read bytes and then convert to numbers.
# I don't really want to use fancy libraries for this.

def _read_idx_images(file_path):
    # what am I going to do here:
    # I will open the file, read the header, and then read all the images as lists of lists.
    images = []
    with open(file_path, 'rb') as f:
        # magic number (we won't use it much), number of images, rows, cols
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # I will just loop through and read each image one by one
        for _ in range(num_images):
            # read rows*cols bytes
            image_bytes = f.read(rows * cols)
            # convert bytes to ints 0..255
            single_image = [list(image_bytes[r*cols:(r+1)*cols]) for r in range(rows)]
            images.append(single_image)
    return images

def _read_idx_labels(file_path):
    labels = []
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        label_bytes = f.read(num_labels)
        for b in label_bytes:
            labels.append(int(b))
    return labels

def load_fashion_mnist_folder(dataset_folder_path):
    # I am going to try and load the train and test splits from the files in the folder.
    train_images_file = os.path.join(dataset_folder_path, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(dataset_folder_path, 'train-labels-idx1-ubyte')
    test_images_file  = os.path.join(dataset_folder_path, 't10k-images-idx3-ubyte')
    test_labels_file  = os.path.join(dataset_folder_path, 't10k-labels-idx1-ubyte')

    train_images = _read_idx_images(train_images_file)
    train_labels = _read_idx_labels(train_labels_file)
    test_images  = _read_idx_images(test_images_file)
    test_labels  = _read_idx_labels(test_labels_file)

    return (train_images, train_labels), (test_images, test_labels)

def filter_to_sneaker_vs_coat(images, labels):
    # I want to only keep examples where the class is coat (4) or sneaker (7)
    # label 4 = "Coat", label 7 = "Sneaker"
    new_images = []
    new_labels = []
    for i, y in enumerate(labels):
        if y == 4 or y == 7:
            new_images.append(images[i])
            # I will map coat->0 and sneaker->1 so that it is easier later
            if y == 4:
                new_labels.append(0)
            else:
                new_labels.append(1)
    return new_images, new_labels

def pad_to_32x32(image28):
    # what am i going to do: add 2 pixels of zero border on each side to make it 32x32
    out = []
    # top pad
    for _ in range(2):
        out.append([0]*32)
    for r in range(28):
        row = [0,0]
        for c in range(28):
            row.append(image28[r][c])
        row.append(0)
        row.append(0)
        out.append(row)
    # bottom pad
    for _ in range(2):
        out.append([0]*32)
    return out

def normalize_0_to_1(image):
    # I think it is ok to just divide by 255.0 to get values between 0 and 1.
    out = []
    for r in range(len(image)):
        new_row = []
        for c in range(len(image[0])):
            new_row.append(image[r][c] / 255.0)
        out.append(new_row)
    return out

def make_minibatches(images, labels, batch_size):
    # I am going to try and split into chunks
    batches = []
    idx = 0
    n = len(images)
    while idx < n:
        end = idx + batch_size
        if end > n:
            end = n
        batch_imgs = []
        batch_lbls = []
        for j in range(idx, end):
            batch_imgs.append(images[j])
            batch_lbls.append(labels[j])
        batches.append((batch_imgs, batch_lbls))
        idx = end
    return batches
