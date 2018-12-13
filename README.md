# MDX CARDS RECOGNITION - Computer Vision Exercise
## Disclaimer
This module was written as part of my submission for Middlesex University's 2018-2019 Advanced Robotics assignment.

## Note
This script is intended for GOOD and not for EVIL
Please do not use in any negative context, at least not without prior consent from
the author of this project.

## Requirements
1. Ubuntu 16.04
2. Python
3. Graphviz, Imutils, Keras, Matplotlib, Numpy, OpenCV, Pillow, Pydot, PyQt5, Scipy, Sklearn, Skimage, Tensorflow
4. Camera / Webcam
5. Classic playing cards with only FOUR suits (Clubs, Diamonds, Hearts, and Spades).
   Special cards are not included.

## Author
Name: Chibuike Okpaluba

Portfolio: https://www.chibuikepraise.com/

Email: co607@live.mdx.ac.uk

Subject: MDX Cards Advanced Robotics Projects 2018

Repo: https://github.com/chibike/mdx_cards_recognition

# Description
This repo/project contains python scripts written to recognize four of the cards suits in a standard card deck. The programs can receive an input from a camera or folder and reliably identify the suits of the card placed in front of it (one at a time).

The recognition process is based on a preprocess-identify computing approach using several CNNs to correctly classify the presented card.

The CNN models used are:
1. ShallowNet,
2. LeNet, and
3. MiniVGGNet.

During the research for this assignment, a KNN based neural network (nn) was used but has not been included in this version.

These models were created based on the description in the pyimagesearch deeplearning Book by Dr. Adrian Rosebrock.

### Based on the Rubric
###### General Quality of Performed Task

The trained models are capable of recognizing the different card suits, and their performance was greatly improved by combining several filters using factors such as axes ratios, minimum and maximum area, and the mean probability for the classified contours.

The combination of these techniques (including the thresholding method) improves its overall resistance to noise (light, etc.). However, the proposed system would perform better in a controlled environment.

###### Correctness of Implementation

The GUI and No GUI applications have been written to handle common bugs/errors (user or built-in) that might occur during execution. 

Furthermore, both applications provide other features such as
1. The option to choose the CNN model used,
2. The option to train the CNN models,
3. The ability to save images directly from the GUI application (by pressing the S key)
4. The option to run the script on a folder containing image samples, etc.

###### Problem Solving and Code Design

The structure of this repo/project is described below.

workspace: Contains the datasets (train, testing, and samples) used and the scripts

workspace/scripts: Contains all the scripts used in the project

Description of relevant scripts.

```
no_app.py    # Simplified version of the GUI application, recommended for scrutiny or detailed understanding of this project
use_app.py   # The main script in the project, launches the GUI to be used to test the models using your webcam

training/*   # Use these scripts to train the model on your dataset
             # To use your trained dataset replace prediction/[model_name]/model/[model_name]_cards.hdf5
             # Ensure that your trained model has the same model_name
```

As mentioned above this project used CNNs.

###### User Experience

There are two ways of launching this project.
1. Using the GUI, which is recommended for live demonstrations, and
2. Running without the GUI on a test folder. This is the recommended starting point if you wish to study how the card's suit is detected.


###### Code Clarity and Modularity, Documentation

Well...., what do I say. In general, I am not a fan of comments, I appreciate GOOD comments but I prefer using appropriate (variables, functions, etc.) names and structure to aid understanding. I have added comments where  I felt it necessary to do so, however, feel free to contact me if you have any questions or bug fixes :).

# Version
v1.0

# How to Run
### Using App
Recommended for running live demonstrations

```
python workspace/scripts/use_app.py
```

### Without App
Recommended for scrutiny -> Be Nice :)

```
python workspace/scripts/no_app.py --model LeNet --dataset workspace/datasets/test_no_app/
```

### Training
Get your hands dirty -> Train the models yourself

```
# To train the ShallowNet model        -- fastest to train
python workspace/scripts/training/shallownet/shallownet_cards.py -d workspace/datasets/preprocessed_cards/cards_dataset

# To train the LeNet model
python workspace/scripts/training/lenet/lenet_cards.py -d workspace/datasets/preprocessed_cards/cards_dataset

# To train the MiniVGGNet model        -- slowest to train
python workspace/scripts/training/minivggnet/minivggnet_cards.py -d workspace/datasets/preprocessed_cards/cards_dataset
```

# Installation (Software) (pip3 is recommeded)
```
sudo apt-get install python3-pip python-pip python3-tk

pip install --user numpy scipy matplotlib jupyter pandas sympy nose imutils pyqt5 Pillow scikit-learn scikit-image
pip install --user opencv-python

pip install --user tensorflow
python -c "import tensorflow as tf; tf.enable_eager_execution();print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
pip install --user keras

sudo apt-get install graphviz

pip install --user graphviz==0.5.2
pip install --user pydot-ng==1.0.0
pip install --user pydot
```

# Questions and Answers
For more information please email the author of this project
