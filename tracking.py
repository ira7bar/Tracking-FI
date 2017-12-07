import cv2
from matplotlib import pylab as plt
import numpy as np
from matplotlib import patches

N = 112
path = "Images/"

def read_frames():
    frames = []
    for i in range(N):
        full_path = path + str(i+1).zfill(3) + '.png'
        f = cv2.imread(full_path)
        frames.append(f)
    return frames

def rgb_to_histogram(rgbImage):
    img_q=rgbImage/16

    redChannel = img_q[:,:, 0]
    greenChannel = img_q[:,:, 1]
    blueChannel = img_q[:,:, 2]

    countsR = cv2.calcHist([redChannel], [0], None, [16], [0, 16])
    countsG = cv2.calcHist([greenChannel], [0], None, [16], [0, 16])
    countsB = cv2.calcHist([blueChannel], [0], None, [16], [0, 16])
    allThree = np.concatenate([countsR, countsG, countsB]).T

    hist_sum = sum(sum(allThree))
    allThree/=hist_sum

    # print rgbImage.shape[0]*rgbImage.shape[1]*rgbImage.shape[2]
    # print sum(sum(allThree))
    # print countsR.shape, countsG.shape, countsB.shape, allThree

    return allThree

def hist_dist(p, q):
    return np.exp(20*np.sum(np.square(p*q)))


def patch_dist(I1,I2):
    p = rgb_to_histogram(I1)
    q = rgb_to_histogram(I2)
    return hist_dist(p,q)


def get_patch(I,h,w,r_cnt,c_cnt):
    top = r_cnt - h/2
    bottom = r_cnt + h/2
    left = c_cnt - w/2
    right = c_cnt + w/2
    patch = I[top:bottom,left:right,:]
    # print h, w, patch.shape
    return patch




frames = read_frames()
f = frames[0]
# plt.imshow(f)
# plt.show()

p1 = get_patch(f,130,50,134,297)
p2 = get_patch(f,130,50,136,300)
p3 = get_patch(f,130,50,290,240)

print patch_dist(p1,p2)
print patch_dist(p2,p1)

print patch_dist(p1,p3)
print patch_dist(p2,p3)

