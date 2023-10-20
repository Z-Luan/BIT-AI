import cv2
import numpy as np

# 一点小tips: 图像没办法完全重构是因为 CV2.Subtract 和 CV2.add 没有互为逆运算

GP_NUM = 6
LP_NUM = GP_NUM - 1
apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')
a = np.ones((256,256))

apple_orange = np.hstack((apple[:, :256], orange[:, 256:]))

apple_copy = apple.copy()
GP_apple = [apple_copy]

for i in range(GP_NUM - 1):
    apple_copy = cv2.pyrDown(apple_copy)
    GP_apple.append(apple_copy)

orange_copy = orange.copy()
GP_orange = [orange_copy]
for i in range(GP_NUM - 1):
    orange_copy = cv2.pyrDown(orange_copy)
    GP_orange.append(orange_copy)

LP_apple = [] 
for i in range(LP_NUM, 0, -1):
    gaussian_expanded = cv2.pyrUp(GP_apple[i])
    laplacian = np.subtract(GP_apple[i-1] , gaussian_expanded)
    LP_apple.append(laplacian)

LP_orange = []
for i in range(LP_NUM, 0, -1):
    gaussian_expanded = cv2.pyrUp(GP_orange[i])
    laplacian = np.subtract(GP_orange[i - 1] , gaussian_expanded)
    LP_orange.append(laplacian)

GP_apple_top = GP_apple[GP_NUM - 1]
GP_orange_top = GP_orange[GP_NUM - 1]
cols, rows, ch = GP_apple_top.shape
GP_apple_orange_top = np.hstack((GP_apple_top[:, 0:int(cols/2)], GP_orange_top[:, int(cols/2):]))

LP_apple_orange = []
for apple_lp, orange_lp in zip(LP_apple, LP_orange):
    cols, rows, ch = apple_lp.shape
    laplacian = np.hstack((apple_lp[:, 0:int(cols/2)], orange_lp[:, int(cols/2):]))
    LP_apple_orange.append(laplacian)

apple_orange_reconstruct = GP_apple_orange_top
for i in range(0, LP_NUM):
    apple_orange_reconstruct = cv2.pyrUp(apple_orange_reconstruct)
    apple_orange_reconstruct = np.add(LP_apple_orange[i] , apple_orange_reconstruct)

cv2.namedWindow('apple_orange_reconstruct', cv2.WINDOW_AUTOSIZE)
cv2.imshow("apple_orange_reconstruct", apple_orange_reconstruct)
# cv2.imshow("apple_orange_direct", apple_orange)
cv2.waitKey(0)
# cv2.destroyAllWindows()