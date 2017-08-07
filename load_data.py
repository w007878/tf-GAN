import os
import tensorflow as tf
import numpy as np
import scipy.io
import cv2
# import matplotlib.pyplot as plt

SVHN_DIR = 'data/SVHN/train_32x32.mat'

def load_SVHN():
    mat = scipy.io.loadmat(SVHN_DIR)
    data = mat['X']
    label = mat['y']
    data = data.transpose([3, 0, 1, 2])
    print(data.shape)

    ans = np.zeros((len(label), 10))
    for i in range(len(label)):
        ans[i, label[i][0] % 10] = 1
        
    return data, ans

#     
# def visual_data(n, m, data, label):
#     for i in range(n * m):
#         print(i)
#         plt.subplot(n, m, i + 1)
#         plt.title(np.argmax(label[i]))
#         plt.imshow(data[i])
#         plt.axis('off')
#     print type(data)
#     plt.show()
    
def cv2_save(n, m, data, file_path=None):
    data = data.reshape((n, m, 32, 32, 3))
    data = data.transpose([0, 2, 1, 3, 4])
    data = data.reshape(n * 32, m * 32, 3)
    image = np.zeros((n * 32, m * 32, 3))
    
    image[:, :, 0] = data[:, :, 2]
    image[:, :, 1] = data[:, :, 1]
    image[:, :, 2] = data[:, :, 0]
    
    # print(image)
    if file_path == None:
        image = image
        cv2.imshow('image', image)
        cv2.waitKey(0)
    else:
        cv2.imwrite((file_path * 256).astype(np.int), image)
    
def test():
    data, label = load_SVHN()
    visual_data(5, 10, data[1:51], label[1:51])
    cv2_save(file_path='tmp.png', n=10, m=10, data=data[0:100])
    print(data.shape)

# test()