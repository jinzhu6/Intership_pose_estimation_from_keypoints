import numpy as np
import cv2
import utils
import os
import argparse
from util_classes import Model, Template, Store
from utils import *
from PIL import Image
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection


def parse_args():
    desc = "Python implementation of the post processing for pose estimation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--image_name', type=str, default='', help='the name of the image')

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert os.path.exists(args.image_name)
    except:
        print('image not found')
        return None

    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    # loading the cad model
    cad = Model()
    cad.load_model()

    # loading dict
    dict = Template(cad)


    # read heatmap and detect maximal responses
    heatmap = readHM(args.image_name, 8)
    [W_hp, score] = findWMax(heatmap);
    lens_f = 319.4593
    lens_f_rescale = lens_f / 640.0 * 64.0
    W_hp[0] = W_hp[0] + 15.013 / 640.0*64.0
    W_hp[1] = W_hp[1] - 64.8108 / 640*64.0
    W_hp_norm = np.ones([3,len(W_hp[0])])
    W_hp_norm[0] = W_hp[0] - 32.0 / lens_f_rescale
    W_hp_norm[1] = W_hp[1] - 32.0 / lens_f_rescale

    # pose estimation weak perspective
    opt_wp = PoseFromKpts_WP(W_hp, dict, weight=score, verb=True,  lam=1, tol=1e-10)

    lens_f_cam = lens_f_rescale * 4
    K_cam = [[lens_f_cam, 0, 128],[0, lens_f_cam, 128],[0, 0, 1]]

    # we use cv2 to read the image to use the cv2 function later
    img = cv2.imread(args.image_name)

    # crop image
    center = [128, 128]
    scale = 1.28
    cropImage(img,center,scale)
    img_crop = cv2.resize(img,(200,200))/255.0
    # weak perpective
    S_wp = np.dot(opt_wp.R,opt_wp.S)
    S_wp[0] += opt_wp.T[0]
    S_wp[1] += opt_wp.T[1]

    # computation of the polygon
    [model_wp, _, _, _] = fullShape(S_wp, cad)

    print(np.std(model_wp.vertices,0))

    mesh2d_wp = np.transpose(model_wp.vertices[:,0:2])*200/heatmap.shape[1]
    # adding the camera parameters
    mesh2d_wp[0] += -15.013
    mesh2d_wp[1] += 64.8108
    mesh2d_wp[0] /= 3.2

    # computation of the sum of the heatmap
    response = np.sum(heatmap,2)

    max_value = np.amax(response)
    min_value = np.amin(response)

    response = (response - min_value)/ (max_value - min_value)

    cmap = plt.get_cmap('jet')
    mapIm = np.delete(cv2.resize(cmap(response),(200,200)),3,2)

    imgToShow = 0.5*mapIm + img_crop*0.5

    print(mesh2d_wp.shape)

    polygon = Polygon(np.transpose(mesh2d_wp))

    plt.figure()
    #plt.imshow(imgToShow)
    print(cad.vertices.shape)

    plt.plot(cad.vertices[:,0], cad.vertices[:,1])
    plt.imshow(imgToShow)
    plt.show()




if __name__ == '__main__':
    main()