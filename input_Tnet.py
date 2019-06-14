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
import PN_tools

def Input_Tnet(num_points, input_points):
    a = Convolution1D(64, 1, activation='relu',input_shape=(num_points, 3))(input_points)
    a = BatchNormalization()(a)
    a = Convolution1D(128, 1, activation='relu')(a)
    a = BatchNormalization()(a)
    a = Convolution1D(1024, 1, activation='relu')(a)
    a = BatchNormalization()(a)
    a = MaxPooling1D(pool_size=num_points)(a)
    a = Dense(512, activation='relu')(a)
    a = BatchNormalization()(a)
    a = Dense(256, activation='relu')(a)
    a = BatchNormalization()(a)
    a = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(a)
    input_T = Reshape((3, 3))(a)
    return input_T