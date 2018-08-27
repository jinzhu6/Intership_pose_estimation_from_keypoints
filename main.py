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
    parser.add_argument('--image_name', type=str, default='./images_test/val_01_00_000006.bmp', help='the name of the image')
    parser.add_argument('--v', type=bool, default=False, help='verbose mode')
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

    # R is the rotation matrix and T the translation matrix for the full perspective
    [img_crop, mapIm, mesh2d_wp, mesh2d_fp, R, T] = mesh_kpts(args.image_name, verbosity=args.v)
    img_with_keypoints = 0.5 * mapIm + img_crop * 0.5

    # configuration of the figure

    #fig = plt.figure('Figure')
    #
    #ax1 = fig.add_subplot(141)
    #ax1.axes.get_xaxis().set_visible(False)
    #ax1.axes.get_yaxis().set_visible(False)
    #ax1.set_title('image')
    #ax1.imshow(img_crop)
    #
    #ax2 = fig.add_subplot(142)
    #ax2.axes.get_xaxis().set_visible(False)
    #ax2.axes.get_yaxis().set_visible(False)
    #ax2.set_title('heatmap')
    #ax2.imshow(img_with_keypoints)
    #
    #ax7 = fig.add_subplot(143)
    #ax7.axes.get_xaxis().set_visible(False)
    #ax7.axes.get_yaxis().set_visible(False)
    #ax7.set_title('cad fp')
    #ax7.imshow(img_crop)
    #polygon = Polygon(mesh2d_wp, linewidth=1, edgecolor='g', facecolor='none')
    #ax7.add_patch(polygon)
    #
    #ax8 = fig.add_subplot(144)
    #ax8.axes.get_xaxis().set_visible(False)
    #ax8.axes.get_yaxis().set_visible(False)
    #ax8.set_title('heatmap and cad fp')
    #ax8.imshow(img_with_keypoints)
    #polygon = Polygon(mesh2d_fp, linewidth=1, edgecolor='g', facecolor='none')
    #ax8.add_patch(polygon)
    #
    #
    ##fig.savefig('./image_rapport/Figure_11.png',bbox_inches='tight')
    #
    #
    #fig.subplots_adjust(wspace=0)
    #
    #
    #plt.show()

    return  0


if __name__ == '__main__':
    main()