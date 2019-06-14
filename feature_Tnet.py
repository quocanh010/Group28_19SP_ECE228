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

def forward_net(input_T, input_points, num_points):
    a = Lambda(PN_tools.mat_mul, arguments={'B': input_T})(input_points)
    a = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(a)
    a = BatchNormalization()(a)
    a = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(a)
    return BatchNormalization()(a)
    
    
def feature_Tnet(g, num_points):
    # feature transform net
    a = Convolution1D(64, 1, activation='relu')(g)
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
    a = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(a)
    feature_T = Reshape((64, 64))(a)
    return feature_T

def forward_net_v2(feature_T, a):
    a = Lambda(PN_tools.mat_mul, arguments={'B': feature_T})(a)
    a = Convolution1D(64, 1, activation='relu')(a)
    a = BatchNormalization()(a)
    a = Convolution1D(128, 1, activation='relu')(a)
    a = BatchNormalization()(a)
    a = Convolution1D(1024, 1, activation='relu')(a)
    return BatchNormalization()(a)