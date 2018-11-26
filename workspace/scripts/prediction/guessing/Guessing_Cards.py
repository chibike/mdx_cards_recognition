#!/usr/bin/env python

import random


class SuitDetector(object):
    def __init__(self):
        # define class labels
        self.class_labels = ["diamonds", "hearts", "spades", "three_sisters"]

    def predict(self, image):
        return random.choice(self.class_labels), (1.0 / len(self.class_labels))

def test():
    s = SuitDetector()


if __name__ == '__main__':
    test()