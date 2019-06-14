
import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
import h5py


def mat_mul(A, B):
    return tf.matmul(A, B)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


# number of points in each sample
num_points = 2048

# number of categories
k = 40

def load_data():

    # load train points and labels
    path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(path, "Prepdata")
    filenames = [d for d in os.listdir(train_path)]
    print(train_path)
    print(filenames)
    train_points = None
    train_labels = None
    for d in filenames:
        if d not in ('.ipynb_checkpoints'):
            cur_points, cur_labels = load_h5(os.path.join(train_path, d))
            cur_points = cur_points.reshape(1, -1, 3)
            cur_labels = cur_labels.reshape(1, -1)
            if train_labels is None or train_points is None:
                train_labels = cur_labels
                train_points = cur_points
            else:
                train_labels = np.hstack((train_labels, cur_labels))
                train_points = np.hstack((train_points, cur_points))
    train_points_r = train_points.reshape(-1, num_points, 3)
    train_labels_r = train_labels.reshape(-1, 1)

    # load test points and labels
    test_path = os.path.join(path, "Prepdata_test")
    filenames = [d for d in os.listdir(test_path)]
    print(test_path)
    print(filenames)
    test_points = None
    test_labels = None
    for d in filenames:
        if d not in ('.ipynb_checkpoints'):
            cur_points, cur_labels = load_h5(os.path.join(test_path, d))
            cur_points = cur_points.reshape(1, -1, 3)
            cur_labels = cur_labels.reshape(1, -1)
            if test_labels is None or test_points is None:
                test_labels = cur_labels
                test_points = cur_points
            else:
                test_labels = np.hstack((test_labels, cur_labels))
                test_points = np.hstack((test_points, cur_points))
    test_points_r = test_points.reshape(-1, num_points, 3)
    test_labels_r = test_labels.reshape(-1, 1)


    # label to categorical
    Y_train = np_utils.to_categorical(train_labels_r, k)
    Y_test = np_utils.to_categorical(test_labels_r, k)
    return train_points_r, test_points_r, Y_train, Y_test

    