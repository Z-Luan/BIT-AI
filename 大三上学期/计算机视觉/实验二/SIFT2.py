import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = 'Minion1.jpg'
imgname2 = 'Minion3.jpg'

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

img1 = cv2.resize(img1, (int(img1.shape[1] / 4), int(img1.shape[0] / 4)))
img2 = cv2.resize(img2, (int(img2.shape[1] / 4), int(img2.shape[0] / 4)))
# cv2.imshow('img1', img1)
# cv2.waitKey(0)
# cv2.imshow('img2', img2)
# cv2.waitKey(0)

img_merge = np.hstack((img1, img2))
# cv2.imshow('img_merge', img_merge)
# cv2.waitKey(0)

b1, g1 ,r1 =cv2.split(img1)
b2, g2 ,r2 =cv2.split(img2)

img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img_gray1', img_merge)
# cv2.waitKey(0)

img_gray_merge = np.hstack((img_gray1, img_gray2))
# cv2.imshow('img_gray_merge', img_gray_merge)
# cv2.waitKey(0)

sift = cv2.SIFT_create()

KeyPoint1, Descriptor1 = sift.detectAndCompute(img_gray1,None)
KeyPoint2, Descriptor2 = sift.detectAndCompute(img_gray2,None)
# KeyPoint1, Descriptor1 = sift.detectAndCompute(g1,None)
# KeyPoint2, Descriptor2 = sift.detectAndCompute(g2,None)
# KeyPoint1, Descriptor1 = sift.detectAndCompute(img1,None)
# KeyPoint2, Descriptor2 = sift.detectAndCompute(img2,None)

img_keypoint1 = cv2.drawKeypoints(img_gray1,KeyPoint1,img_gray1,color=(0, 255, 255)) 
img_keypoint2 = cv2.drawKeypoints(img_gray2,KeyPoint2,img_gray2,color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_keypoint_merge = np.hstack((img_keypoint1, img_keypoint2))
# cv2.imshow("img_keypoint_merge", img_keypoint_merge)
# cv2.waitKey(0)

BF = cv2.BFMatcher()
Match = BF.knnMatch(Descriptor1,Descriptor2, k=2)
# print(Matches)
# print(len(Matches))

Good_Match = []
for i, j in Match:
    if i.distance < 0.75 * j.distance:
        Good_Match.append([i])

# Match_img = cv2.drawMatchesKnn(img_gray1,KeyPoint1,img_gray2,KeyPoint2,Good_Match,None,flags=2)
Match_img = cv2.drawMatchesKnn(img1,KeyPoint1,img2,KeyPoint2,Good_Match,None,flags=2)
cv2.imshow("Match_img", Match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
