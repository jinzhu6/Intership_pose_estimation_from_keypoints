import numpy as np
import cv2
import utils
import os
import argparse



def parse_args():
    desc = "Python implementation of the post processing for pose estimation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--image_name', type=str, default='', help='the name of the image')

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert os.path.exists(image_name)
    except:
        print('image not found')
        return None

    return args


def main():
    args = parse_args()
    if args is None:
        exit()





if __name__ == '__main__':
    main()