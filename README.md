# Multi-Class Logistic Regression Model with Pytorch on MNIST Dataset

Modelling a multi-class logistic regression model with pytorch to predict the digit(0-9) in an image of 28X28 pixels.
This model is implemented from Scratch in Python. The best test accuracy of my model is around 91%.

Libraries we used:
- Numpy
- Matplotlib
- Torch
- Torchvision
- tqdm

Used both basic equations and library functions for the indicator vector generator function, softmax function, gradient function.
The gradient function is G = -(e(y) – softmax(theta*X))^(T) – X^(T).
