import cv2
from matplotlib import pylab as plt
import numpy as np
from matplotlib import patches

N = 112
path = "Images/"


def normalized_hist(I, width, height, xc, yc):
    pass


def predict_kalman():
    print 'predict!'

def read_frames():
    frames = []
    for i in range(N):
        full_path = path + str(i+1).zfill(3) + '.png'
        f = plt.imread(full_path)
        frames.append(f)
    return frames


frames = read_frames()
normalized_hist(frames[0])