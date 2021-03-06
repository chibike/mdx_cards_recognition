#!/usr/bin/env python

#python shallownet_cards.py -d ../../../datasets/preprocessed_cards/cards_dataset

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,"{}/../../".format(currentdir))

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.ImageToArrayPreprocessor import ImageToArrayPreprocessor
from preprocessing.SimplePreprocessor import SimplePreprocessor
from data_loading.SimpleDatasetLoader import SimpleDatasetLoader
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

NAME      = "shallownet_cards"
DIR_NAME  = "{}_results".format(NAME)
IM_NAME   = "{}/{}.png".format(DIR_NAME, NAME)
DATA_NAME = "{}/{}.hdf5".format(DIR_NAME, NAME)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=False, default=".", help="path to output folder")
args = vars(ap.parse_args())

print ("[INFO] verifying output directory...")
if not os.path.exists(args["output"]):
    raise ValueError("Invalid output path received")

print("[INFO] loading images")
image_paths = list(paths.list_images(args["dataset"]))

n = 32
input_size = n*n*3
sp = SimplePreprocessor(n, n)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print ("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print ("[INFO] training network...")
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=100, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=["diamonds", "hearts", "spades", "three_sisters"]))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epock #")
plt.ylabel("Loss/Accuracy")
plt.legend()


print ("[INFO] setting up storage location ....")

DIR_NAME  = os.path.join(args["output"], DIR_NAME)

try:
    os.rmdir(DIR_NAME)
    print ("[INFO] removed old directory")
except OSError:
    print ("[INFO] could not remove old directory")

try:
    os.mkdir(DIR_NAME)
    print ("[INFO] created new directory")
except OSError:
    print ("[INFO] could not create new directory")

print ("[INFO] saving network....")

plt.savefig(IM_NAME)
model.save(DATA_NAME)

print ("[INFO] your network has been saved!")