#!/usr/bin/env python

# py -2 shallownet.py


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
            width   : The width of the input images that will be used
            height  : The height of the input images
            depth   : The number of channels in the image
            classes : The total number of classes that should be learnt
        '''

        model = Sequential()
        
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
        
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
