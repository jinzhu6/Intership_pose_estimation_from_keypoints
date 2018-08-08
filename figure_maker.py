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

    fig1 = plt.figure('Figure1')

    ax1 = fig1.add_subplot(111)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    #ax1.set_title('image')
    ax1.imshow(img_crop)

    fig2 = plt.figure('Figure2')

    ax2 = fig2.add_subplot(111)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    #ax2.set_title('heatmap')
    ax2.imshow(img_with_keypoints)

    fig3 = plt.figure('Figure3')

    ax3 = fig3.add_subplot(111)
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    #ax3.set_title('cad fp')
    ax3.imshow(img_crop)
    polygon = Polygon(mesh2d_fp, linewidth=1, edgecolor='g', facecolor='none')
    ax3.add_patch(polygon)

    fig4 = plt.figure('Figure4')

    ax4 = fig4.add_subplot(111)
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)
    #ax4.set_title('heatmap and cad fp')
    ax4.imshow(img_with_keypoints)
    polygon = Polygon(mesh2d_fp, linewidth=1, edgecolor='g', facecolor='none')
    ax4.add_patch(polygon)

    fig5 = plt.figure('Figure5')

    ax5 = fig5.add_subplot(111)
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)
    # ax5.set_title('cad fp')
    ax5.imshow(img_crop)
    polygon = Polygon(mesh2d_wp, linewidth=1, edgecolor='g', facecolor='none')
    ax5.add_patch(polygon)

    fig6 = plt.figure('Figure6')

    ax6 = fig6.add_subplot(111)
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)
    # ax6.set_title('heatmap and cad fp')
    ax6.imshow(img_with_keypoints)
    polygon = Polygon(mesh2d_wp, linewidth=1, edgecolor='g', facecolor='none')
    ax6.add_patch(polygon)


    fig1.savefig('./image_rapport/Figure_11.png', bbox_inches='tight')
    fig2.savefig('./image_rapport/Figure_21.png', bbox_inches='tight')
    fig3.savefig('./image_rapport/Figure_31.png', bbox_inches='tight')
    fig4.savefig('./image_rapport/Figure_41.png', bbox_inches='tight')
    fig5.savefig('./image_rapport/Figure_51.png', bbox_inches='tight')
    fig6.savefig('./image_rapport/Figure_61.png', bbox_inches='tight')



    return  0


if __name__ == '__main__':
    main()