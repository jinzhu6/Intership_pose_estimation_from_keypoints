import numpy as np
import cv2
import utils
import os
import argparse
from CAD import CAD
from utils import *


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
    dict = CAD()
    dict.load_cad()

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
    output_wp = PoseFromKps_WP(W_hp,dict,'weight',score,'verb',true,'lam',1,'tol',1e-10)




if __name__ == '__main__':
    main()