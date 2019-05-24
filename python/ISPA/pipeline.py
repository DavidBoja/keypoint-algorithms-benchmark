

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
from matplotlib.patches import ConnectionPatch
from pprint import pprint
from sklearn.metrics import average_precision_score
from random import sample

from own_implementations import HarrisMataHarris, ShiTomasi
from utils import *
from evaluatons import patchVerification, imageMatching, patchRetrieval
from visualizations import taskEvaluation

all_detectors = {'sift':cv2.xfeatures2d.SIFT_create,
                 'surf':cv2.xfeatures2d.SURF_create,
                 'ORB':cv2.ORB_create,
                 'fast':cv2.FastFeatureDetector_create,
                 'brisk':cv2.BRISK_create,
                 'harris':HarrisMataHarris,
                 'shi_tomasi':ShiTomasi,
                 'kaze':cv2.KAZE_create}

all_descriptors = {'sift':cv2.xfeatures2d.SIFT_create,
                   'surf':cv2.xfeatures2d.SURF_create,
                   'ORB':cv2.ORB_create,
                   'brisk':cv2.BRISK_create,
                   'freak':cv2.xfeatures2d.FREAK_create,
                   'kaze':cv2.KAZE_create}


# Set Variables
detector_name = 'sift'
descriptor_name = 'sift'
project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
dataset_path = project_root + '/hpatches-sequences-release/*'
n = 100


# Obtain keypoints
createKeypoints(detector_name, descriptor_name, dataset_path)
removeUncommonPoints(detector_name, descriptor_name, dataset_path)

# Evaluations
list_of_APs_pV, mAP_pV = patchVerification(detector_name, descriptor_name, n, dataset_path)
list_of_APs_iM, mAP_for_iteration_iM, nr_of_iterations_iM = imageMatching(detector_name,
                                                                          descriptor_name,
                                                                          n,
                                                                          dataset_path)

list_of_APs_pR, mAP_pR = patchRetrieval(detector_name, descriptor_name, n, dataset_path)

# Drawing results
dict_of_APs = dict(zip(detector_name,list_of_APs_pV))
tastEvaluation(dict_of_APs,title_name='Patch Verification')

dict_of_APs = dict(zip(detector_name,list_of_APs_iM))
tastEvaluation(dict_of_APs,title_name='Image matching')

dict_of_APs = dict(zip(detector_name,list_of_APs_pR))
tastEvaluation(dict_of_APs,title_name='Patch Retrieval')
