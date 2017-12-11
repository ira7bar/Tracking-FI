###### This code works ##########
import cv2
from matplotlib import pylab as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

N = 112
path = "Images/"



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
    allThree /= hist_sum

    # print rgbImage.shape[0]*rgbImage.shape[1]*rgbImage.shape[2]
    # print sum(sum(allThree))
    # print countsR.shape, countsG.shape, countsB.shape,
    # print allThree

    return allThree

def hist_dist(p, q):
    return np.sqrt(1-np.sum(np.sqrt(p*q)))
    # return np.exp(-20*np.sum(np.sqrt(p*q)))
    # return np.exp(-20*np.sum(np.sqrt(p*q)))
    # return -np.log2(20*np.sum(np.square(p*q)))


def patch_dist(I1,I2):
    p = rgb_to_histogram(I1)
    q = rgb_to_histogram(I2)
    #return hist_dist(p, q) #/ np.sqrt(hist_dist(q, q) * hist_dist(p, p))
    return hist_dist(p, q)
    #return hist_dist(p,q)/np.sqrt(hist_dist(q,q)*hist_dist(p,p))


def get_patch(I,h,w,r_cnt,c_cnt):
    top = r_cnt - h/2
    bottom = r_cnt + h/2
    left = c_cnt - w/2
    right = c_cnt + w/2
    patch = I[top:bottom, left:right, :]
    # print h, w, patch.shape
    return patch

class Window(object):
    def __init__(self,rc=(0,0),h=50,w=50):
        self.rc = rc
        self.h = h
        self.w = w


class Kalman(object):
    def __init__(self,rc,vrc):
        self.frames = self.read_frames()
        self.X = np.array([rc[0],rc[1],vrc[0],vrc[1]])


    def predict(self):
        pass

    def update(self):
        pass

    def read_frames(self):
        frames = []
        for i in range(N):
            full_path = path + str(i + 1).zfill(3) + '.png'
            f = cv2.imread(full_path)
            frames.append(f)
        return frames

    def plot_window(self,f,w,color = 'r'):
        ''' input: center, uses corners'''
        fig,ax = plt.subplots(1)
        ax.imshow(f)
        r = w.rc[0]-w.h/2
        c = w.rc[1]-w.w/2
        rect = patches.Rectangle((c,r),w.w,w.h,linewidth=1,edgecolor=color,facecolor='none')
        #TODO cv2.rectangle
        ax.add_patch(rect)
        plt.show()

    def sliding_window(self, frame, search_window, target_patch, search_shift, dist_func):
        query_patch_shape = target_patch.shape
        rcorn_bias = search_window.rc[0] - search_window.h / 2
        ccorn_bias = search_window.rc[1] - search_window.w / 2
        print "query path shape", query_patch_shape
        print "r bias", rcorn_bias
        print "c bias", ccorn_bias
        rcorn_vector = np.arange(rcorn_bias, rcorn_bias + search_window.h + search_shift[0] - query_patch_shape[0], search_shift[0])
        ccorn_vector = np.arange(ccorn_bias, ccorn_bias + search_window.w + search_shift[1] - query_patch_shape[1], search_shift[1])
        print "r_vector centers"
        print rcorn_vector+query_patch_shape[0] / 2
        print "c_vector centers"
        print ccorn_vector+query_patch_shape[1] / 2
        distances = np.full((len(rcorn_vector), len(ccorn_vector)), np.nan)
        print "Distances shape", distances.shape
        i = 0
        for r in rcorn_vector:
            r_center = r + query_patch_shape[0] / 2
            j = 0
            for c in ccorn_vector:
                c_center = c + query_patch_shape[1] / 2
                patch = get_patch(frame, query_patch_shape[0], query_patch_shape[1], r_center, c_center)
                distances[i, j] = dist_func(target_patch, patch)
                w = Window((r_center, c_center), query_patch_shape[0], query_patch_shape[1])
                k.plot_window(frame, w, color='b')
                print "at row = {}, col = {}, the distance is {}".format(r_center, c_center, distances[i, j])
                j += 1
            i+= 1
        distances = distances
        min_val = np.amin(distances)
        min_row = rcorn_vector[np.where(distances == min_val)[0][0]] + query_patch_shape[0] / 2
        min_col = ccorn_vector[np.where(distances == min_val)[1][0]] + query_patch_shape[1] / 2
        #min_row = np.where(distances == min_val)[0][0] + rcorn_bias + query_patch_shape[0] / 2
        #min_col = np.where(distances == min_val)[1][0] + ccorn_bias + query_patch_shape[1] / 2

        return (min_row, min_col), min_val, distances









k = Kalman((134,297),(0,0))

f = k.frames[0]
w = Window((134,297),121,49)
k.plot_window(f,w)

f = k.frames[1]
w = Window((133,295),121,51)
k.plot_window(f,w)
w2 = Window((134, 293), 161, 161)
k.plot_window(f, w2, color = 'y')

my_search_window = Window((134, 293),161, 161)
target_image  = get_patch(f,121,49,134, 293)
search_shift = (10,10)
my_dist_func = patch_dist
point_ind, minimum_value, distances_matrix  = k.sliding_window(f, my_search_window , target_image, search_shift,
                                                               my_dist_func)
print "point ind", point_ind
print "min val", minimum_value

w3 = Window((point_ind[0], point_ind[1]), 121, 49)
k.plot_window(f, w3, color = 'y')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.05)
x = range(distances_matrix.shape[1])
y = range(distances_matrix.shape[0])
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, distances_matrix)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


#
# frames = read_frames()
# f = frames[0]
# plt.imshow(f)
# plt.show()

p1 = get_patch(f,133,50,134,295)
p2 = get_patch(f,130,50,136,300)
p3 = get_patch(f,130,50,290,240)

print patch_dist(p1,p1)
print patch_dist(p1,p3)
print patch_dist(p2,p3)