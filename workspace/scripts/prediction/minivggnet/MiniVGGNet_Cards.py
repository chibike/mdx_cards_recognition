#!/usr/bin/env python

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,"/home/mdxcards/Desktop/Project/mdx_cards_recognition/workspace/scripts/")

from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from preprocessing.SimplePreprocessor import SimplePreprocessor
from data_loading.SimpleDatasetLoader import SimpleDatasetLoader
from keras.models import load_model
import numpy as np
import cv2


class SuitDetector(object):
    def __init__(self):
        # define class labels
        self.class_labels = ["diamonds", "hearts", "spades", "clubs"]

        # define preprocessors
        self.pp = [SimplePreprocessor(32, 32), ImageToArrayPreprocessor()]

        # load model
        self.model = load_model("{}/minivggnet_cards.hdf5".format(currentdir))

    def predict(self, image):
        for p in self.pp:
            image = p.preprocess(image)

        # convert image array to float (0. -> 1.)
        data = np.array([image])
        data = data.astype("float") / 255.0

        # make prediction
        pred = self.model.predict(data, batch_size=32)

        # assign label
        label = self.class_labels[pred.argmax(axis=1)[0]]
        probability = pred[0][pred.argmax(axis=1)[0]]

        return label, probability



def test():
    s = SuitDetector()


if __name__ == '__main__':
    test()