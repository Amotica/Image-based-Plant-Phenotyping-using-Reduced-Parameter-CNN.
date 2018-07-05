# Image-based-Plant-Phenotyping-using-Reduced-Parameter-CNN.
Segmentation is the core of most plant phenotyping applications. Current state-of-the-art plant phenotyping applications rely on deep Convolutional Neural Networks (CNNs). Phenotyping applications relying on these deep CNNs are often difficult if not impossible to deploy on limited-resource devices.
This work is a first step to moving plant phenotyping applications in-field and on low-cost devices with limited resources. 
This work re-architect four baseline deep neural networks (creating what we term "Lite CNN") by reducing their parameters whilst making them deeper to avoid overfitting. 

# Minimum dependencies required to use these codes:
Python 3.6.1

Keras 2.0.6

Tensorflow 1.3.0

Numpy 1.13.3

Pillow 5.1.0

Opencv 3.2.0

# Running the codes:
Use train.py to train the baseline models and train_lite.py can be used to train the "Lite" models

Use evaluate.py to evaluate the baseline models and evaluate_lite.py to evaluate the "Lite" models.

Most network parameters can be changed in the parameter file.

All CNN models are contained in the Models folder

Dataset not included but can be downloaded from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
