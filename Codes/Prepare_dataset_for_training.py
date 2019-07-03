############################  Import Libraries  ###############################
###############################################################################

import numpy as np
import imutils
import cv2
from operator import itemgetter
import math
import os
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import shutil


data_folder = "./Data/All/"
extraction_folder = "./Extracted/"
troughs_folder = "./Extracted/Troughs/"
vein_folder = "./Extracted/Vein_Images/"
bounding_box_folder = "./Extracted/Bounding_box/"
pos_vein_folder = "./Extracted/Vein_Images/pos_vein_img/"
neg_vein_folder = "./Extracted/Vein_Images/neg_vein_img"
