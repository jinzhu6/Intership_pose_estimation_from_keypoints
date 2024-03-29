import autograd.numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import utils
import os
import time
import sys
from util_classes import Store, Output, Model, Template
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import TrustRegions
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection




def readHM(filepath, M):
    '''
    read the output of the neural network and return the heatmaps
    we use plt to read the image because it is simpler than cv2
    :param filepath : path to the file
    :param M : number of keypoints
    : : 3D array[64,64,M] with the raw keypoints
    '''

    HM = np.zeros([64,64,M])
    for i in range(M):
        hm_name = filepath[0:len(filepath)-4] + '_{:02d}'.format(i+1) + filepath[len(filepath)-4:len(filepath)]
        #print(hm_name)
        HM[:,:,i] = cv2.imread(hm_name)[:,:,0]

    return HM/255.0

def cropImage(image,center,scale):
    '''
    crop the image and rezise it as an 200 by 200 image
    :param image : the image you want to resize
    :param center : the center of the image form which you want to resize the image
    :param scale : the cropping scale
    :return : the cropped and resized image
    '''

    w = int(200*scale)
    h = int(w)
    x = int(center[0] - w/2)
    y = int(center[1] - h/2)
    im = cv2.copyMakeBorder( image, w, w, h, h, cv2.BORDER_CONSTANT)
    im1 = im[w:x+2*w,h:y+2*h]
    im1 = cv2.resize(im1, (200, 200), interpolation=cv2.INTER_CUBIC)

    return im1

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
        W_max[0,i] = y[0] + 1
        W_max[1,i] = x[0] + 1 # +1 to compare with matlab

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

    [U, S, V] = np.linalg.svd(A,4)

    Q = U[:,0:2]
    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = np.array([ np.trace(Ti) , Ti[1,0] - Ti[0,1] ])
        G = G + np.outer(gi,np.transpose(gi)) # it is really import to use outer and not dot !!! test yourself

    [U1, S1, V1] = np.linalg.svd(G)

    G = np.zeros((2,2))
    for i in range(d):
        Ai = X[3*i:3*(i+1),:]
        Ti = np.dot(np.transpose(Q),Ai)
        gi = np.array([ Ti[0,0]-Ti[1,1] , Ti[0,1]+Ti[1,0] ])
        G = G + np.outer(gi, np.transpose(gi))

    [U2, S2, V2] = np.linalg.svd(G,4)

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
    # creation of the store object, for now it is not usefull be may contribute the the improvement of the code
    store = Store()
    ####################################################################################################################
    def cost(X):
        '''
        cost function of the manifold, the cost is trace(E'*D*E)/(2*N) with E = A*X - B or store.E
        :param X : vector
        :param store: a Store oject to store some information
        :return : [f,score] f is the score, store is the Store object
        '''
        if store.E is None:
            store.E = np.dot(A,np.transpose(X))-B

        E = store.E
        f = np.trace(np.dot(np.transpose(E),np.dot(D,E)))/2

        return f

    def grad(X):
        '''
        grad function of the manifold, the gradient is the reimannian gradient computed with the manifold
        :param X : vector
        :param store : a Store oject to store some information
        :return : [g,store] g is the gradient and store is the Store object
        '''
        if store.E is None:
            _ = cost(X)

        E = store.E
        # compute the euclidean gradient of the cost with the rotations R and the cloud A
        egrad = np.dot(At,np.dot(D,E))
        # transform this euclidean gradient into the Riemmanian gradient
        g = manifold.egrad2rgrad(np.transpose(X),egrad)
        store.egrad = egrad

        return np.array(g)
    ####################################################################################################################

    # setup the problem structure with manifold M and cost and grad function
    problem = Problem(manifold=manifold, cost=cost, verbosity=0)

    # setup the trust region algorithm to solve the problem
    TR = TrustRegions(maxiter=10)

    # solve the problem
    X = TR.solve(problem,X0)

    return np.transpose(X) # return R = X'

def estimateC_weighted(W, R, B, D, lam):
    '''
    :param W : the heatmap
    :param R : the rotation matrix
    :param B : the base matrix
    :param D : the weight
    :param lam : lam value used to simplify some results
    :return : C0
    '''
    p = len(W[0])
    k = int(B.shape[0]/3)
    d = np.diag(D)
    D = np.zeros((2*p,2*p))
    eps = sys.float_info.epsilon

    for i in range(p):
        D[2*i, 2*i] = d[i];
        D[2*i+1, 2*i+1] = d[i];

    # next we work on the linear system y = X*C
    y = W.flatten() # vectorized W
    X = np.zeros((2*p,k)) # each colomn is a rotated Bk

    for i in range(k):
        RBi = np.dot(R,B[3*i:3*(i+1),:])
        X[:,i] = RBi.flatten()


    # we want to calculate C = pinv(X'*D*X+lam*eye(size(X,2)))*X'*D*y and then C = C'
    A = np.dot(np.dot(np.transpose(X),D),X) + lam*np.eye(X.shape[1])
    tol = max(A.shape) * np.linalg.norm(A,np.inf) * eps
    C = np.dot(np.dot(np.linalg.pinv(A),np.dot(np.transpose(X),D)),y)

    return np.transpose(C)

def PoseFromKpts_WP(W, dict, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    compute the pose with weak perspective
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return ; return a Output object containing many informations
    '''

    # data size
    B = np.copy(dict.mu)  # B is the base
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
    for iter in range(1000):

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
        PrimRes = np.linalg.norm(M-Z) / (np.linalg.norm(Z0)+eps)
        DualRes = mu*np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0)+eps)

        # show output
        if verb:
            print('Iter = ', iter, ' ; PrimRes = ',PrimRes, '; DualRes = ', DualRes,' ; mu = ', '{:08.6f}'.format(mu), '\n')

        # check convergente
        if PrimRes < tol and DualRes < tol:
            break
        else:
            if PrimRes > 10 * DualRes:
                mu = 2 * mu;
            elif DualRes > 10 * PrimRes:
                mu = mu / 2;
            else:
                pass

    # end iteration

    [R, C] = syncRot(M)

    if np.sum(np.abs(R)) == 0:
        R = np.eye(3)

    R = R[0:2,:]
    S = np.dot(np.kron(C,np.eye(3)),B)

    # iteration, part 2
    fval = np.inf

    for iter in range(1000):
        T = np.sum(np.dot((W-np.dot(R,S)),D), 1) / (np.sum(D)+eps) # T = sum((W-R*S)*D, 1) / (sum(D)+eps)
        W2fit = np.copy(W)
        W2fit[0] -= T[0]
        W2fit[1] -= T[1]

        # update rotation
        R = np.transpose(estimateR_weighted(S, W2fit, D, R))


        # update shape
        if len(pc) == 0:
            C0 = estimateC_weighted(W2fit, R, B, D, 1e-3)[0]
            S = C0*B
        else:
            W_1 = W2fit - np.dot(np.dot(R , np.kron(C, eye(3))) , pc)
            C0 = estimateC_weighted(W_1, R, B, D, 1e-3)
            W_2 = W2fit - np.dot(np.dot(R , C0) , B)
            C = estimateC_weighted(W_2, R, pc, D, lam)
            S = np.dot(C0,B) + np.dot(np.kron(C,np.eye(3)),pc)

        fvaltml = fval
        # fval = 0.5*norm((W2fit-R*S)*sqrt(D),'fro')^2 + 0.5*norm(C)^2;
        fval = 0.5*np.linalg.norm(np.dot(W2fit-np.dot(R,S),np.sqrt(D)),'fro')**2 + 0.5*np.linalg.norm(C)**2

        # show output
        if verb:
            print('Iter = ', iter, 'fval = ', fval)

        # check convergence
        if np.abs(fval-fvaltml) / (fvaltml+eps) < tol:
            break

    # end iteration
    R2 = np.zeros((3,3))
    R2[0,:] = R[0,:]
    R2[1, :] = R[1, :]
    R2[2,:] = np.cross(R[0,:],R[1, :])
    output = Output(S=S, M=M, R=R2, C=C ,C0=C0, T=T, fval=fval)

    return output

def PoseFromKpts_FP(W, dict, R0=None, weight=None, verb=True, lam=1, tol=1e-10):
    '''
    compute the pose with full perspective
    solve in ||W*diag(Z)-R*S-T||^2 + ||C||^2 with S = C1*B1+...+Cn*Bn, Z denotes the depth of points
    :param W: the maximal responses in the headmap
    :param dict: the cad model
    :param varargin: other variables
    :return ; return a Output object containing many informations
    '''
    # data size
    mu = np.copy(dict.mu)  # B is the base
    pc = np.copy(dict.pc)
    R = np.copy(R0)
    # setting values
    if weight is None:
        D = np.eye(p)
    else:
        D = np.diag(weight)

    # centralize basis
    meanmu = np.mean(mu, 1)
    for i in range(mu.shape[0]):
        mu[i] -= meanmu[i]

    # initialization
    eps = sys.float_info.epsilon
    S = mu
    T = np.mean(W,1) * np.mean(np.std(np.dot(R[0:2,:],S),1)) / (np.mean(np.std(W,1))+eps)
    C = 0

    fval = np.inf

    # iteration
    for iter in range(1000):

        # update the depth of Z
        Z = np.dot(R,S)
        for j in range(3):
            Z[j] += T[j]

        Z = np.sum(W * Z,0) / (np.sum(W**2, 0)+eps)

        # update R and T by aligning S to W*diag(Z)
        Sp = np.dot(W, np.diag(Z))
        T = np.sum(np.dot(Sp-np.dot(R,S),D), 1) / (np.sum(np.diag(D))+eps)
        St = Sp
        for j in range(Sp.shape[0]):
            St[j] -= T[j]

        [U, _, V] = np.linalg.svd(np.dot(St,np.dot(D,np.transpose(S))))
        R = np.dot(np.dot(U,np.diag([1 , 1 , np.linalg.det(np.dot(U,V))])),V)

        fvaltml = fval
        fval = np.linalg.norm(np.dot(St-np.dot(R,S),np.sqrt(D)), 'fro')**2 + lam*np.linalg.norm(C)**2

        # show output
        if verb:
            print('Iter = ',iter, 'fval = ',fval)

        # check convergence
        if np.abs(fval-fvaltml) / (fvaltml+eps) < tol:
            break


    output = Output(S=S, R=R, C=C, T=T, Z=Z, fval=fval)

    return output

def findRotation(S1,S2):
    '''
    find the rotation matrix between S1 and S2
    :param S1 : matrix 1
    :param S2 : matrix 2
    :return : R, the rotation R*S1 = S2
    '''
    [f,p] = S1.shape
    f = int(f/3)
    S1 = np.reshape(S1,(3,f*p))
    S2 = np.reshape(S2,(3,f*p))
    R = np.dot(S1,np.transpose(S2))
    # /!\ the matlab svd computes R = USV' and the python svd computes R = USV
    [U, _, V] = np.linalg.svd(R)
    R = np.dot(U,V)
    R = np.dot(U,  np.dot(np.diag([1.0,1.0,np.linalg.det(R)]), V))

    return R

def fullShape(S1,model):
    '''
    creates the new model besed on the S
    :param S1 : the matrix S (I do not know what it is exactly
    :param model : an object of the class Model
    :return : the new object Model, and some other information and transformation matrix
    '''
    eps = sys.float_info.epsilon

    # normalization of S
    S2 = np.copy(np.transpose(model.kp))
    T1 = np.mean(S1,1)
    T2 = np.mean(S2,1)
    for i in range(len(T1)):
        S1[i] -= T1[i]
        S2[i] -= T2[i]

    R = findRotation(S1,S2)
    S2 = np.dot(R,S2)
    w = np.trace(np.dot(np.transpose(S1), S2))/(np.trace(np.dot(np.transpose(S2), S2))+eps);
    T = T1 - w*np.dot(R,T2)

    vertices = np.transpose(model.vertices)

    for i in range(len(T)):
        vertices[i] = vertices[i] - T2[i]

    vertices = w*np.dot(R,vertices)

    for i in range(len(T)):
        vertices[i] = vertices[i] + T1[i]

    model_new = model.copy()
    model_new.vertices = np.transpose(vertices)
    model_new.nb_vertices = len(vertices)

    return [model_new,w,R,T]

def get_transform(center,scale,res):
    '''
    no idea what this think does
    :param center:
    :param scale:
    :param res:
    :return:
    '''
    h = 200*scale
    t = np.eye(3)
    t[0, 0] = res[1] / h
    t[1, 1] = res[0] / h
    t[0, 2] = res[1] * (-center[0] / h + 0.5)
    t[1, 2] = res[0] * (-center[0] / h + 0.5)
    t[2, 2] = 1

    return t

def transformHG(pt,center,scale,res,invert):
    '''
     no fucking idea what this think does
    :param pt:
    :param center:
    :param scale:
    :param res:
    :param invert:
    :return:
    '''
    t = get_transform(center,scale,res)

    if invert:
        t = np.linalg.inv(t)

    new_pt = np.zeros(pt.shape)
    new_pt[0] = pt[0]
    new_pt[1] = pt[1] - 0.5
    new_pt[2] = np.ones(new_pt[2].shape)

    new_pt = np.dot(t,new_pt)

    return new_pt[0:2]



def mesh_kpts(image_name,verbosity=True, lam=1, tol=1e-10):
    '''
    take the image name and return the croped image, the image of the heatmap and the meshs of the cad model
    :param image_name : the path/name of the image
    :param verbosity : print or not some indermediate results
    :param lam : usualy 1
    :param tol : a little value used in convergence tests
    :return : [cropped_image, heatmap, meshs]
    '''
    # loading the cad model
    cad = Model()
    cad.load_model()

    # loading dict
    dict = Template(cad)

    # read heatmap and detect maximal responses
    heatmap = readHM(image_name, 8)
    [W_hp, score] = findWMax(heatmap);
    print(W_hp,"\n")
    print(score,"\n")
    lens_f = 319.4593
    lens_f_rescale = lens_f / 640.0 * 64.0
    W_hp[0] = W_hp[0] + 15.013 / 640.0 * 64.0
    W_hp[1] = W_hp[1] - 64.8108 / 640 * 64.0
    W_hp_norm = np.ones([3, len(W_hp[0])])
    W_hp_norm[0] = (W_hp[0] - 32.0) / lens_f_rescale
    W_hp_norm[1] = (W_hp[1] - 32.0) / lens_f_rescale

    # pose estimation weak perspective
    opt_wp = PoseFromKpts_WP(W_hp, dict, weight=score, verb=verbosity, lam=lam, tol=tol)

    # pose estimation full perspective
    opt_fp = PoseFromKpts_FP(W_hp_norm, dict, R0=opt_wp.R, weight=score, verb=verbosity, lam=1, tol=1e-10);

    lens_f_cam = lens_f_rescale * 4
    K_cam = [[lens_f_cam, 0, 128], [0, lens_f_cam, 128], [0, 0, 1]]

    # we use cv2 to read the image to use the cv2 function later
    img = cv2.imread(image_name)

    # crop image
    center = [128, 128]
    scale = 1.28
    cropImage(img, center, scale)
    img_crop = cv2.resize(img, (200, 200)) / 255.0

    # weak perspective
    S_wp = np.dot(opt_wp.R, opt_wp.S)
    S_wp[0] += opt_wp.T[0]
    S_wp[1] += opt_wp.T[1]

    # full perspective
    S_fp = np.dot(opt_fp.R,opt_fp.S)
    for i in range(S_fp.shape[0]):
        S_fp[i] += opt_fp.T[i]

    # computation of the polygon weak perspective
    [model_wp, _, _, _] = fullShape(S_wp, cad)
    mesh2d_wp = np.transpose(model_wp.vertices[:, 0:2]) * 200 / heatmap.shape[1]
    # adding the camera parameters
    mesh2d_wp[0] += -15.013 / 3.2
    mesh2d_wp[1] += 64.8108 / 3.2

    # computation of the polygon full perspective
    [model_fp, _, _, _] = fullShape(S_fp, cad)
    mesh2d_fp = np.dot(K_cam,np.transpose(model_fp.vertices))
    # adding camera parameters
    mesh2d_fp[0] /= mesh2d_fp[2]
    mesh2d_fp[1] /= mesh2d_fp[2]
    mesh2d_fp = transformHG(mesh2d_fp, center, scale, heatmap.shape[0:2], False) * 200 / heatmap.shape[1]
    mesh2d_fp[0] += -15.013 / 3.2
    mesh2d_fp[1] += 64.8108 / 3.2

    # computation of the sum of the heatmap
    response = np.sum(heatmap, 2)

    max_value = np.amax(response)
    min_value = np.amin(response)

    response = (response - min_value) / (max_value - min_value)

    cmap = plt.get_cmap('jet')
    mapIm = np.delete(cv2.resize(cmap(response), (200, 200)), 3, 2)

    return [img_crop, mapIm, np.transpose(mesh2d_wp), np.transpose(mesh2d_fp), opt_fp.R, opt_fp.T]



