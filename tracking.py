import cv2
from matplotlib import pylab as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

N = 112
path = "Images/"



def rgb_to_histogram(rgbImage):
    img_q=rgbImage/8

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
    return np.exp(-20*np.sum(np.sqrt(p*q)))
    # return 1-np.sum(np.sqrt(p*q))
    # return -np.log2(20*np.sum(np.square(p*q)))


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

class Window(object):
    def __init__(self,rc=(0,0),h=50,w=50):
        self.rc = rc
        self.h = h
        self.w = w


class Kalman(object):
    def __init__(self,rc,vrc):
        self.frames = self.read_frames()
        self.X = np.array([rc[0],rc[1],vrc[0],vrc[1]])
        # self.cov = np.array([[]])

        self.F = np.array([[1,1/20.0],[0,1]])


    def predict(self):
        self.X_predicted = np.dot(self.F,self.X)
    # #     for frame in self.frames:



    def update(self):
        pass

    def read_frames(self):
        frames = []
        for i in range(N):
            full_path = path + str(i + 1).zfill(3) + '.png'
            f = cv2.imread(full_path)
            frames.append(f)
        return frames

    def plot_window(self,f,w):
        fig,ax = plt.subplots(1)
        ax.imshow(f)
        r = w.rc[0]-w.h/2
        c = w.rc[1]-w.w/2
        rect = patches.Rectangle((c,r),w.w,w.h,linewidth=1,edgecolor='r',facecolor='none')
        #TODO cv2.rectangle
        ax.add_patch(rect)
        plt.show(block=False)
        plt.pause(1)
        plt.close()
        # plt.close()



    def sliding_window(self,frame,search_w,test_patch_size,target_patch,search_shift):
        test_patch_h = test_patch_size[0]
        test_patch_w = test_patch_size[1]
        r_bias = search_w.rc[0] - search_w.h/2
        c_bias = search_w.rc[1] - search_w.w/2
        r_vector = np.arange(r_bias,r_bias+search_w.h+search_shift[0]-test_patch_h,search_shift[0])
        c_vector = np.arange(c_bias,c_bias+search_w.w+search_shift[1]-test_patch_w,search_shift[1])
        print len(r_vector)
        print len(c_vector)
        distances = np.full((len(r_vector),len(c_vector)),np.inf)
        print distances.shape
        i=0
        for r in r_vector:
            j=0
            r_center = r + test_patch_h/2
            for c in c_vector:
                c_center = c + test_patch_w / 2
                patch = get_patch(frame,test_patch_h,test_patch_w,r_center,c_center)
                distances[i,j] = patch_dist(target_patch,patch)
                j += 1
            i += 1

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # x = y = np.arange(-3.0, 3.0, 0.05)
        # x = range(distances.shape[1])
        # y = range(distances.shape[0])
        # X, Y = np.meshgrid(x, y)
        # ax.plot_surface(X, Y, distances)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

        min_val = np.amin(distances)
        min_row_i = np.where(distances == min_val)[0][0]
        min_col_j = np.where(distances == min_val)[1][0]
        print distances

        min_row = r_vector[min_row_i] + r_bias
        min_col = c_vector[min_col_j] + c_bias

        # best_patch = get_patch(frame, test_patch_h, test_patch_w, min_row, min_col)
        # plt.imshow(best_patch)
        # plt.show()

        return [min_row, min_col, min_val]


    # def running(self):










k = Kalman((134,297),(0,0))

# ref_h = 100
ref_h = 115
ref_w = 35

# w_ref = Window((134,297),ref_h,ref_w) # manual input
w_ref = Window((135,297),ref_h,ref_w) # manual input
f_ref = k.frames[0]


# k.plot_window(f,w_ref)

search_w = Window((165, 284), f_ref.shape[0] // 2, f_ref.shape[1] // 2)
target_patch = get_patch(f_ref, w_ref.h, w_ref.w, w_ref.rc[0], w_ref.rc[1])
plt.imshow(target_patch)
plt.show()

for i_frame in k.frames:


    # def sliding_window(self, frame, search_w, test_patch_size, target_patch, search_shift):
    # search_w = Window((172,284),250,400)
    # search_w = Window((172,284),242,392)

    min_row, min_col, min_val =  k.sliding_window(i_frame,search_w,(w_ref.h,w_ref.w),target_patch,(5,5))
    print min_row, min_col, min_val
    best_window=Window((min_row,min_col),ref_h,ref_w)

    k.plot_window(i_frame, best_window)




#
# frames = read_frames()
# f = frames[0]
# plt.imshow(f)
# plt.show()

# p1 = get_patch(f,130,50,134,297)
# p2 = get_patch(f,130,50,136,300)
# p3 = get_patch(f,130,50,290,240)
#
# print patch_dist(p1,p2)
# print patch_dist(p2,p1)
#
# print patch_dist(p1,p3)
# print patch_dist(p2,p3)

