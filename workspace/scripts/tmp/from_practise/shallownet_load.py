#!/usr/bin/env python


#python shallownet_load.py -d ../datasets_2/cards_dataset -m shallownet_cards_results/shallownet_cards.hdf5

from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from preprocessing.SimplePreprocessor import SimplePreprocessor
from datasets.SimpleDatasetLoader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

class_labels = ["diamonds", "hearts", "spades", "three_sisters"]

print("[INFO] sampling images")
image_paths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(image_paths), size=(10,))
image_paths = image_paths[idxs]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths)
data = data.astype("float") / 255.0

print ("[INFO] loading model...")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    # cv2.putText(image, , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print("[INFO] Label: {}".format(class_labels[preds[i]]))
    cv2.imshow("image", image)
    cv2.waitKey(0)