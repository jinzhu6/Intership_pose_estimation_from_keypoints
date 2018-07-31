import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import utils
import os
import time





def readHM(filepath, M):
    '''
    :param filepath: path to the file
    :param M: number of keypoints
    :return: 3D array[64,64,M] with the raw keypoints
    '''

    HM = np.zeros([64,64,M])
    for i in range(M):
        hm_name = filepath[0:30] + '_{:02d}'.format(i+1) + '.bmp'
        #print(hm_name)
        HM[:,:,i] = plt.imread(hm_name)

    return HM/255.0


def findWMax(hm):
    '''
    :param hm: the heatmap given by readHM
    :return: [W_max, score] where W_max is a array containing the coordinates of the maximum and score is the value of the maximum
    '''
    p = hm.shape[2]
    W_max = np.zeros([2,p])
    score = np.zeros(p)

    for i in range(p):
        score[i] = np.amax(hm[:,:,i])
        (x,y) = np.where(hm[:,:,i]==score[i])
        W_max[0,i] = x[0]
        W_max[1,i] = y[0]

    return [W_max, score]


def PoseFromKpts_WP(W, dict, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return: TODO : document the return
    '''

    # data size
    B = dict.mu
    pc = dict.pc
    [k,p] =  B.shape
    k /= 3

    # setting values
    if weight is None:
        D = np.eye(p)
    else:
        D = np.diag(weight)

    alpha = 1

    # centralize basis





    return 0 # TODO : implement this function