import os
import tensorflow as tf
import numpy as np
import scipy.io
import cv2
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('data/SVHN/train_32x32.mat')
data = mat['X']
label = mat['y']

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.title(label[i][0])
    plt.imshow(data[..., i])
    plt.axis('off')
print type(data)
plt.show()