#!/usr/bin/env python


import os
import cv2
import numpy as np


class SimpleDatasetLoader(object):
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, image_paths, verbose=-1):
        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print ("[INFO] processed {}/{}".format(i+1, len(image_paths)))
        
        return (np.array(data), np.array(labels))