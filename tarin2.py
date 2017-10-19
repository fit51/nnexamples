#%%
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import tensorflow as tf

from IPython.display import display, Image
from scipy import ndimage, misc
import h5py

# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline

#%%

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

    dataset = load_dataset("not_mnist_dataset.h5")
    print(dataset["train_set"].shape)
    print(dataset["train_labels"].shape)
    print(dataset["valid_set"].shape)
    print(dataset["valid_labels"].shape)
    print(dataset["test_set"].shape)
    print(dataset["test_labels"].shape)

#%%
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

#%%
#reformat
def reformat(dataset, labels):
    dataset = dataset.reshape(-1, image_size, image_size, num_channels).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(dataset["train_set"], dataset["train_labels"])
valid_dataset, valid_labels = reformat(dataset["valid_set"], dataset["valid_labels"])
test_dataset, test_labels = reformat(dataset["test_set"], dataset["test_labels"])
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
    
#%%
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#%%
m1 = [[1, 2, 3],
    [1, 2, 3],
[1, 2, 3]]
np.argmax(predictions, 1)

#%%
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
