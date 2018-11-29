# mdx_cards_recognition
This repository was created as part of the necessary submission for a 2018/2019 Robotics assignment

# Disclaimer:
This module was written as part of my submission for Middlesex Univeristy's 2018-2019 Advanced Robotics assigment.

# Note
This script in intended for GOOD and not for EVIL
Please do not use in any negative context, at least not without prior consent from
the author of this project.

# Requirements
1. Ubuntu 16.04
2. Python
3. Graphviz, Imutils, Keras, Matplotlib, Numpy, OpenCV, Pillow, Pydot, PyQt5, Scipy, Sklearn, Skimage, Tensorflow
4. Camera/Webcam
5. Classic playing cards with only FOUR suits (Clubs, Diamonds, Hearts, and Spades).
   Special cards are not included.

# Author
Email: co607@live.mdx.ac.uk
Subject: MDX Cards Advanded Robotics Projects 2018

# Description
Nothing for now...

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
python workspace/scripts/train.py
```

# Installation (Software)
```
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