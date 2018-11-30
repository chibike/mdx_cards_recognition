#!/usr/bin/env python
r'''
----------------------------- No App ----------------------------------
Implementation of The Cards Classification Code without a GUI

Please not that this script uses "Python 3" and the following additional libaries

    matplotlib, imutils, numpy, scipy, sklearn, keras, Pillow, and tensorflow

    Most of the other scripts have been written by "Chibuike Okpaluba", please read LICENSE.txt for more information.


DESCRIPTION

    This scripts load all images in a given folder, and displays their predicted suits


HOW TO USE

    run: python no_app.py --model LeNet --dataset ../datasets/test_no_app/

    expectation: To windows contained the original and processed image should be launched


FOR MORE INFORMATION

    Contact: co607@live.mdx.ac.uk
    Subject: MDX Cards Advanded Robotics Projects 2018

'''

from __future__ import division

from matplotlib import pyplot as plt
from imutils import paths
import numpy as np
import cv2

from prediction.guessing   import Guessing_Cards
from prediction.shallownet import ShallowNet_Cards
from prediction.lenet      import LeNet_Cards
from prediction.minivggnet import MiniVGGNet_Cards

import support_functions
import argparse

print("[INFO] parsing your arguments")
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",   required=False, default="LeNet", help="model name")
ap.add_argument("-d", "--dataset", required=True,  help="path to input images")
args = vars(ap.parse_args())

models = {
    "Guessing"   : Guessing_Cards,
    "ShallowNet" : ShallowNet_Cards,
    "LeNet"      : LeNet_Cards,
    "MiniVGGNet" : MiniVGGNet_Cards
}

print("[INFO] sampling images")
image_paths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(image_paths), size=(len(image_paths),)) # randomize the input data
image_paths = image_paths[idxs]

print("[INFO] loading selected model")
model_name = args["model"]

if not model_name in models.keys():
    raise ValueError("The --model parameter must be one of the following [{}]".format(", ".join(list(models.keys()))))

prediction_model = models[model_name].SuitDetector()

print("[INFO] showing images")
for image_path in image_paths:
    img = cv2.imread(image_path)
    (processed_img, results), gray_th = support_functions.process_image(image_path, prediction_model)

    print () # print blank line
    print ("[INFO] results for {0} = (suit: {1}, prob: {3:.3f})".format(model_name, *results))
    print ("[INFO] PRESS SPACE to load the next image, Q to exit.")

    cv2.imshow("original vs processed", support_functions.combine_images(img, processed_img))
    k = cv2.waitKey(0)

    if k in [27, ord('q')]:
        print ("[INFO] exiting...")
        break

cv2.destroyAllWindows()
exit()

