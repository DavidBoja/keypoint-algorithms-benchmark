

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
import itertools

from own_implementations import HarrisMataHarris, ShiTomasi
from utils import *
from evaluations import patchVerification, imageMatching, patchRetrieval
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

project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
dataset_path = project_root + '/hpatches-sequences-release/*'
n = 100
pV_results = {}
iM_results = {}
pR_results = {}

det_des_combinations = list(itertools.product(list(all_detectors.keys()),
                                              list(all_descriptors.keys()) )
                                              )

for dett, dess in det_des_combinations:
    print('Computing combination {}:{}'.format(dett,dess))
    # Set Variables
    detector_name = dett
    descriptor_name = dess


    # Obtain keypoints
    createKeypoints(detector_name, descriptor_name, dataset_path)
    removeUncommonPoints(detector_name, descriptor_name, dataset_path)

    # Evaluations
    list_of_APs_pV = patchVerification(detector_name, descriptor_name, n, dataset_path)
    pV_results[detector_name + '_' + descriptor_name] = list_of_APs_pV

    list_of_all_APs_iM, list_of_mAPs_iM = imageMatching(detector_name,
                                                        descriptor_name,
                                                        n,
                                                        dataset_path)
    iM_results[detector_name + '_' + descriptor_name] = list_of_mAPs_iM

    list_of_all_APs_pR, list_of_mAPs_pR = patchRetrieval(detector_name, descriptor_name, n, dataset_path)
    pR_results[detector_name + '_' + descriptor_name] = list_of_mAPs_pR

    deleteAllKeypoints(dataset_path)

# Drawing results
taskEvaluation(pV_results,title_name='Patch Verification')

taskEvaluation(iM_results,title_name='Image matching')

taskEvaluation(pR_results,title_name='Patch Retrieval')
