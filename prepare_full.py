#%%
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from IPython.display import display, Image
from scipy import ndimage, misc
import h5py

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline

#%%
base_dir='/home/kodratyuk/PyProjects/gooNN/'
small_dir=base_dir + "notMNIST_small"
big_dir=base_dir + "notMNIST_large"
image_size = 28
dataset_size = 200000
valid_size = 10000
train_size = 10000

#%%
number = 2132
ldir = base_dir + "notMNIST_small/A/"
images = os.listdir(ldir)
image = misc.imread(ldir + images[number % len(images)], True)
image = image / 255.0
print(image.shape)
print(image)
plt.imshow(image)
display(image)

#%%

def load_letter(folder, print_process=True):
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
  print("Operating on: ", folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = misc.imread(image_file, True)
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, : , :] = image_data
      num_images += 1
      if num_images % 5000 == 0:
        print(num_images, "loaded")
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  dataset = dataset[:num_images, :, :]
  print(folder, "Done! ", "Loaded: ", dataset.shape[0])
  print("Mean: ", np.mean(dataset))
  print("STD: ", np.std(dataset))
  return dataset

start = time.time()
a_examples = load_letter(big_dir + "/A")
end = time.time()
print("Taken time: ", end - start)
a_examples.shape

def load_image(folder, print_process=True):
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
  print("Operating on: ", folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = misc.imread(image_file, True)
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, : , :] = image_data
      num_images += 1
      if num_images % 5000 == 0:
        print(num_images, "loaded")
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  dataset = dataset[:num_images, :, :]
  print(folder, "Done! ", "Loaded: ", dataset.shape[0])
  print("Mean: ", np.mean(dataset))
  print("STD: ", np.std(dataset))
  return dataset

#%%

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray(shape=(nb_rows, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.float32)
  else:
    dataset = None
    labels = None
  return dataset, labels

def load_data(folder):
  labels = os.listdir(folder)
  num_classes = len(labels)
  current_class = 0
  current_position = 0
  current_size = 0
  num_images = 0
  label_images = []
  for label in labels:
      path = os.path.join(folder, label)
      label_image.append((label, path))
  dataset_size = len(label_images)
  result_dataset, labels_dataset = make_arrays(dataset_size, image_size)
  for label_image in label_images:
    try:
      image_data = misc.imread(image_file, True)
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, : , :] = image_data
      num_images += 1
      if num_images % 5000 == 0:
        print(num_images, "loaded")
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    dataset = load_letter(os.path.join(folder, label))
    current_size = len(images)
    l_data = np.ndarray(len(dataset), dtype=np.float32)
    l_data.fill(current_class)
    end = current_position + size_per_class
    result_dataset[current_position : end] = dataset[:size_per_class]
    labels_dataset[current_position : end] = l_data[:size_per_class]
    current_position += size_per_class
    current_class += 1
  return  result_dataset, labels_dataset

test_dataset, test_labels = load_data(small_dir, train_size)
print(test_dataset.shape, test_labels.shape)
trainValid_dataset, trainValid_labels = load_data(big_dir, dataset_size + valid_size)
print(trainValid_dataset.shape, trainValid_labels.shape)

#%%
def shuffle(dataset, label_dataset):
  permutation = np.random.permutation(dataset.shape[0])
  dataset = dataset[permutation, :, :]
  label_dataset = label_dataset[permutation]
  return dataset, label_dataset

def split_dataset(dataset, label_dataset, valid_size):
  size = len(dataset)
  return dataset[:size - valid_size, :, :], label_dataset[:size - valid_size], dataset[size - valid_size: size, :, :], label_dataset[size - valid_size:size]

shuffled_test_dataset, shuffled_test_labels = shuffle(test_dataset, test_labels)
shuffled_trainValid_dataset, shuffled_trainValid_labels = shuffle(trainValid_dataset, trainValid_labels)
shuffled_train_dataset, shuffled_train_labels, shuffled_valid_dataset, shuffled_valid_labels = split_dataset(shuffled_trainValid_dataset, shuffled_trainValid_labels, valid_size)

#%%
number = 1
plt.imshow(shuffled_valid_dataset[number])
print(shuffled_valid_labels[number])

#%%
#save to h5
h5f = h5py.File('not_mnist_dataset_full.h5', 'w')
h5f.create_dataset("train/dataset", data=shuffled_train_dataset)
h5f.create_dataset("train/labels", data=shuffled_train_labels)
print("Saved" + "train")
h5f.create_dataset("valid/dataset", data=shuffled_valid_dataset)
h5f.create_dataset("valid/labels", data=shuffled_valid_labels)
print("Saved" + "valid")
h5f.create_dataset("test/dataset", data=shuffled_test_dataset)
h5f.create_dataset("test/labels", data=shuffled_test_labels)
print("Saved" + "test")
h5f.close()

#%%
# load dataset

def load_dataset(file):
  h5f = h5py.File(file, 'r')
  dataset = {
    "train_set" : h5f["train/dataset"][:],
    "train_labels" : h5f["train/labels"][:],
    "valid_set" : h5f["valid/dataset"][:],
    "valid_labels" : h5f["valid/labels"][:],
    "test_set" : h5f["test/dataset"][:],
    "test_labels" : h5f["test/labels"][:]
  }
  h5f.close()
  return dataset

dataset = load_dataset("not_mnist_dataset1.h5")
print(dataset["train_set"].shape)
print(dataset["train_labels"].shape)
print(dataset["valid_set"].shape)
print(dataset["valid_labels"].shape)
print(dataset["test_set"].shape)
print(dataset["test_labels"].shape)
number = 123434
print(dataset["train_labels"][number])
plt.imshow(dataset["train_set"][number])