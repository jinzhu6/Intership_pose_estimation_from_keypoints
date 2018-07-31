import numpy as np
import cv2
import utils
import os
import argparse
from CAD import CAD
from utils import *


cad = CAD()
cad.load_cad()

path = './images_test'
imname = path + '/' + 'val_01_00_000000.bmp'

HM = readHM(imname, 8)
findWMax(HM)