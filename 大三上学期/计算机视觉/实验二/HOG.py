from skimage import feature as ft
import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('Minion2.jpg', cv2.IMREAD_GRAYSCALE)
features = ft.hog(img, orientations=6, pixels_per_cell=[20,20], cells_per_block=[2,2], visualize=True)
# print(type(features))
# print(len(features))
# print(features[0])
# print(features[1].shape)
plt.imshow(features[1],cmap=plt.cm.gray)
plt.show()
