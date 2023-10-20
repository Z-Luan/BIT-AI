from traceback import print_tb
import numpy as np
import cv2

def GP(img, Num):
    pyramid = [img]
    for i in range(Num):
        blur_img = cv2.GaussianBlur(pyramid[i], (5,5), 0)
        downsample_img = blur_img[::2, ::2]
        pyramid.append(downsample_img)
    
    return pyramid

def LP(img, Num):
    pyramid = []
    now_img = img
    for i in range(Num):
        blur_img = cv2.GaussianBlur(now_img, (5,5), 0)
        res = now_img - blur_img
        pyramid.append(res)
        now_img = blur_img[::2, ::2]
        if i == Num - 1:
            pyramid.append(now_img)
    
    return pyramid

def Blend(LP1, LP2, MP):

    BlendPyramid = []
    for i in range(len(LP1)):
        blend = MP[i] * LP1[i] + (1 - MP[i]) * LP2[i]
        BlendPyramid.append(blend)

    return BlendPyramid

def Upsample(img):
    shape = list(img.shape)
    shape[0] *= 2
    shape[1] *= 2
    insert = np.zeros(shape)
    insert[::2, ::2] = img 
    blur = 4 * cv2.GaussianBlur(insert, (5,5), 0)
    return blur

def Reconstruct(LP):
    length = len(LP)
    top = LP[-1]

    for i in reversed(range(0, length-1)):
        upsampled = Upsample(top)
        if top.shape[0] == LP[i].shape[0]:
            upsampled = np.delete(upsampled, -1, axis = 0)
        if top.shape[1] == LP[i].shape[1]:
            upsampled = np.delete(upsampled, -1, axis = 1)
        top = upsampled + LP[i]

    return top

def main(img1, img2, mask, Num):
    LP1 = LP(img1, Num)
    LP2 = LP(img2, Num)
    MP = GP(mask, Num)
    blended_pyramid = Blend(LP1, LP2, MP)
    
    return Reconstruct(blended_pyramid)

if __name__ == '__main__':

    orange = cv2.imread('orange.jpg').astype(np.float64)
    apple = cv2.imread('apple.jpg').astype(np.float64)

    mask = np.zeros((orange.shape[0], orange.shape[1], 3)).astype(np.float64)
    mask[:, 0:256, :] = 1
 
    Num = 10

    apple_orange = np.zeros(orange.shape)
    for i in range(3): 
        channel = main(apple[:,:,i], orange[:,:,i], mask[:,:,i], Num)
        apple_orange[:,:,i] = channel
    
    cv2.imshow('apple_orange', apple_orange.astype(np.uint8))
    cv2.waitKey(0)