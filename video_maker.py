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
import matplotlib.animation as manimation


def parse_args():
    desc = "Python implementation of the post processing for pose estimation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--images_name', type=str, default='', help='the name generic of the images')
    parser.add_argument('--nb_image', type=int, default=0, help='the number of images you have')
    parser.add_argument('--v', type=bool, default=False, help='verbose mode')
    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert os.path.exists(args.images_name)
    except:
        print('image not found')
        return None

    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    FFMpegWriter = manimation.writers['pillow']
    metadata = dict(title='demo', artist='Pujolle',
                    comment='')
    writer = FFMpegWriter(fps=30, metadata=metadata)

    with writer.saving(fig, "animation_cosinus.mp4", 100):
        for i in range(args.nb_image+1):
            im_name = args.images_name
            im_name = im_name[0:len(im_name)-10] + '{:06d}'.format(i) + im_name[len(im_name)-4:len(im_name)]

            [img_crop, mapIm, _, mesh2d_fp, _, _] = mesh_kpts(args.images_name, verbosity=args.v)
            img_with_keypoints = 0.5 * mapIm + img_crop * 0.5

            # configuration of the figure

            fig = plt.figure('Figure')

            ax1 = fig.add_subplot(141)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)
            ax1.set_title('image')
            ax1.imshow(img_crop)

            ax2 = fig.add_subplot(142)
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            ax2.set_title('heatmap')
            ax2.imshow(img_with_keypoints)

            ax7 = fig.add_subplot(143)
            ax7.axes.get_xaxis().set_visible(False)
            ax7.axes.get_yaxis().set_visible(False)
            ax7.set_title('cad fp')
            ax7.imshow(img_crop)
            polygon = Polygon(mesh2d_fp, linewidth=1, edgecolor='g', facecolor='none')
            ax7.add_patch(polygon)

            ax8 = fig.add_subplot(144)
            ax8.axes.get_xaxis().set_visible(False)
            ax8.axes.get_yaxis().set_visible(False)
            ax8.set_title('heatmap and cad fp')
            ax8.imshow(img_with_keypoints)
            polygon = Polygon(mesh2d_fp, linewidth=1, edgecolor='g', facecolor='none')
            ax8.add_patch(polygon)

            fig.subplots_adjust(wspace=0)

            writer.grad_frame()




    return  0


if __name__ == '__main__':
    main()