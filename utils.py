import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import utils
import os
import time
import sys




def readHM(filepath, M):
    '''
    read the output of the neural network and return the heatmaps
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
    read the heatmap and return the coordinates of the values of the maximum of the heatmap
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

def prox_2norm(Z,lam):
    '''
    X is a 3 by 2 matrix
    I do not understand what this function does
    '''
    [U,S,V] = np.linalg.svd(Z,)


def PoseFromKpts_WP(W, dict, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    compute the pose with weak perspective
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return: TODO : document the return
    '''

    # data size
    B = np.copy(dict.mu)
    pc = np.copy(dict.pc)
    [k,p] =  B.shape
    k = int(k/3)

    # setting values
    if weight is None:
        D = np.eye(p)
    else:
        D = np.diag(weight)

    alpha = 1

    # centralize basis
    mean = np.mean(B, 1)
    for i in range(3*k):
        B[i] -= mean[i]

    # initialization
    M = np.zeros([2, 3 * k]);
    C = np.zeros([1, k]); # norm of each Xi


    # auxiliary variable for ADMM
    Z = np.copy(M)
    Y = np.copy(M)

    eps = sys.float_info.epsilon
    mu = 1/(np.mean(W)+eps)

    # pre-computing
    BBt = np.matmul(B,np.matmul(D,np.transpose(B)))

    # iteration
    for i in range(1):

        # update translation
        T = np.sum(np.matmul((W-np.matmul(Z,B)),D), 1) / (np.sum(D)+eps) # T = sum((W-Z*B)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update motion matrix Z
        Z0 = np.copy(Z)
        Z = np.matmul( np.matmul(W2fit,np.matmul(D,np.transpose(B))) + mu*M + Y , np.linalg.inv(BBt+mu*np.eye(3*k))) # Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k))

        # update motion matrix M
        Q = Z - Y/mu
        #for i in range(k):




    return 0 # TODO : implement this function