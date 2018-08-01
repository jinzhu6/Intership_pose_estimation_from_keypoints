import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import utils
import os
import time
import sys
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import TrustRegions




def readHM(filepath, M):
    '''
    read the output of the neural network and return the heatmaps
    :param filepath : path to the file
    :param M : number of keypoints
    : : 3D array[64,64,M] with the raw keypoints
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
    :param hm : the heatmap given by readHM
    :return : [W_max, score] where W_max is a array containing the coordinates of the maximum and score is the value of the maximum
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
    This function simplifies Z based on the value of lam and the svd of Z
    :param Z : matrix that need to be simplified
    :param lam : cutting parameter
    :return : [X, normX] this simplified matrix and the its first singular value
    '''

    [U,w,V] = np.linalg.svd(Z) # Z = U*W*V
    if np.sum(w) <= lam:
        w = [0,0]
    elif w[0] - w[1] <=lam:
       w[0] = (np.sum(w) - lam) / 2
       w[1] = w[0]
    else:
        w[0] = w[0] - lam
        w[1] = w[1]

    W = np.zeros(Z.shape)
    W[:len(Z[0]),:len(Z[0])] = np.diag(w)
    X = np.dot(U,np.dot(W,V)) # X = U*W*V
    normX = w[0]

    return [X, normX]

def proj_deformable_approx(X):
    '''
    Ref: A. Del Bue, J. Xavier, L. Agapito, and M. Paladini, "Bilinear
    Factorization via Augmented Lagrange Multipliers (BALM)" ECCV 2010.

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; version 2, June 1991

    USAGE: Y = proj_deformable_approx(X)

    This function projects a generic matrix X of size 3*K x 2 where K is the
    number of basis shapes into the matrix Y that satisfy the manifold
    constraints. This projection is an approximation of the projector
    introduced in: M. Paladini, A. Del Bue, S. M. s, M. Dodig, J. Xavier, and
    L. Agapito, "Factorization for Non-Rigid and Articulated Structure using
    Metric Projections" CVPR 2009. Check the BALM paper, Sec 5.1.

    :param X : the 3*K x 2 affine matrix
    :return : the 3*K x 2 with manifold constraints
    '''

    r = X.shape[0]
    d = int(r / 3)
    A = np.zeros((3,3))

    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        A = A + np.dot(Ai,np.transpose(Ai))

    [U, S, V] = np.linalg.svd(A)
    Q = U[:,0:2]
    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = [ np.trace(Ti) , Ti[1,0] - Ti[0,1] ]
        G = G + np.dot(gi, np.transpose(gi))

    [U1, S1, V1] = np.linalg.svd(G)
    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = [ Ti[0,0]-Ti[1,1] , Ti[0,1]+Ti[1,0] ]
        G = G + np.dot(gi, np.transpose(gi))

    [U2, S2, V2] = np.linalg.svd(G)

    if S1[0] > S2[0]:
        u = U1[:,0]
        R = [[u[0], -u[1]],[u[1], u[0]]]
    else:
        u = U2[:,0]
        R = [[u[0], u[1]], [u[1], -u[0]]]

    Q = np.dot(Q,R)

    Y = np.zeros([d*Q.shape[0],Q.shape[1]])
    L = np.zeros(d)
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        ti = 0.5*np.trace(np.dot(np.transpose(Q),Ai))
        L[i] = ti
        Y[:,2*i:2*(i+1)] = ti*Q


    return [Y, L, Q]

def syncRot(T):
    '''
    returns the rotation matrix of the approximation of the projector created by proj_deformable_approx
    :param T : the motion matrix calculated in PoseFromKpts_WP
    :return : [R,C] the rotation matrix and the values of a sorte of norm that is do not really understand
    '''
    [_, L, Q] = proj_deformable_approx(np.transpose(T))
    s = np.sign(L[np.argmax(np.abs(L))])
    C = s*np.transpose(L)
    R = np.zeros((3,3))
    R[0:2,0:3] = s*np.transpose(Q)
    R[2,0:3] = np.cross(Q[0:3,0],Q[0:3,1])

    return [R,C]

def estimateR_weighted(S,W,D,R0):
    '''
    estimates the update of the rotation matrix for the second part of the iterations
    :param S : do know
    :param W : heatmap
    :param D : weight of the heatmap
    :param R0 : rotation matrix
    :return: R the new rotation matrix
    '''

    A = np.transpose(S)
    B = np.transpose(W)
    X0 = R0[0:2,:]


    [m,n] = A.shape


    p = B.shape[1]

    At = np.zeros([n, m]);
    At =  np.transpose(A)

    # we use the optimization on a Stiefel manifold because R is constrained to be othogonal
    manifold = Stiefel(n,p,1)

    ####################################################################################################################
    def cost(X,store):
        '''
        cost function of the manifold, the cost is trace(E'*D*E)/(2*N) with E = A*X - B or store.E
        :param X : vector
        :param store: a Store oject to store some information
        :return : [f,score] f is the score, store is the Store object
        '''
        if store.E is None:
            store.E = np.dot(A,X)-B

        E = store.E
        f = np.trace(np.dot(np.transpose(E),np.dot(D,E)))/2

        return f
    def grad(X,store):
        '''
        grad function of the manifold, the gradient is the reimannian gradient computed with the manifold
        :param X : vector
        :param store : a Store oject to store some information
        :return : [g,store] g is the gradient and store is the Store object
        '''
        if store.E is None:
            [_, store] = cost(X,store)

        E = store.E
        # compute the euclidean gradient of the cost with the rotations R and the cloud A
        egrad = np.dot(At,np,dot(D,E))
        # transform this euclidean gradient into the Riemmanian gradient
        g = manifold.egrad2rgrad(X,egrad)
        store.egrad = egrad

        return np.array(g)
    ####################################################################################################################

    # setup the problem structure with manifold M and cost and grad function
    problem = Problem(manifold=manifold, cost=cost, grad=grad, verbosity=0)

    # setup the trust region algorithm to solve the problem
    TR = TrustRegions(maxiter=10)

    # solve the problem
    X = TR.solve(problem,X0)

    # R = np.transpose(X)










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
    C = np.zeros(k); # norm of each Xi


    # auxiliary variable for ADMM
    Z = np.copy(M)
    Y = np.copy(M)

    eps = sys.float_info.epsilon
    mu = 1/(np.mean(W)+eps)

    # pre-computing
    BBt = np.dot(B,np.dot(D,np.transpose(B)))

    # iteration
    for iter in range(1):

        # update translation
        T = np.sum(np.dot((W-np.matmul(Z,B)),D), 1) / (np.sum(D)+eps) # T = sum((W-Z*B)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update motion matrix Z
        Z0 = np.copy(Z)
        Z = np.dot( np.dot(W2fit,np.dot(D,np.transpose(B))) + mu*M + Y , np.linalg.inv(BBt+mu*np.eye(3*k))) # Z = (W2fit*D*B'+mu*M+Y)/(BBt+mu*eye(3*k))

        # update motion matrix M
        Q = Z - Y/mu
        for i in range(k):
            [X, normX] = prox_2norm(np.transpose(Q[:,3*i:3*i+3]),alpha/mu)
            M[:, 3*i:3*i+3] = np.transpose(X)
            C[i] = normX

        # update dual variable
        Y = Y + mu*(M-Z)
        PrimRes = np.linalg.norm(M-Z) / (np.linalg.norm(Z0)+eps);
        DualRes = mu*np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0)+eps);

        # show output
        if verb:
            print('Iter = ', iter, ' ; PrimRes = ',PrimRes, '; DualRes = ', DualRes,' ; mu = ', '{:08.6f}'.format(mu), '\n')

        # convergente ?
        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu = 2 * mu;
            elif DualRes > 10 * PrimRes:
                mu = mu / 2;
            else:
                pass

    # fin iteration

    [R, C] = syncRot(M)
    if np.sum(np.abs(R)) == 0:
        R = np.eye(3)

    R = R[0:2,:]
    S = np.dot(np.kron(C,np.eye(3)),B)


    # iteration, part 2
    fval = np.inf

    for iter in range(1):

        # update translation
        T = np.sum(np.dot((W-np.matmul(R,S)),D), 1) / (np.sum(D)+eps) # T = sum((W-R*S)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update rotation
        estimateR_weighted(S, W2fit, D, R)

    return 0 # TODO : implement this function