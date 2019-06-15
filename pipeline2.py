
# PIPELINE2 JE ZA POKRENUTI SKUPINU det+des IN BULK
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import os
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import average_precision_score
from random import sample
import itertools
from pprint import pprint
import json
#from skimage.feature import CENSURE

from own_implementations import *
from utils import *
from evaluations import patchVerification, imageMatching, patchRetrieval
#from visualizations import taskEvaluation

# project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
# dataset_path = project_root + '/hpatches-sequences-release/*'
dataset_path = '/home/dbojanic/hpatches-sequences-release/*'

n = 100
nr_iterations = 5
failed_combinations = []
FILE_EXTENSION = '_zver3_3.txt'

det_des_combinations = list(itertools.product(list(all_detectors.keys()),
                                              list(all_descriptors.keys()) )
                                              )

combinations_to_remove = [(ORB, SIFT),(ORB, SURF),(ORB, BRISK),(ORB, FREAK),(ORB, KAZE),(ORB,AKAZE),(ORB, BRIEF),(ORB, ROOT_SIFT),
                          (SIFT,ORB),(SURF,ORB),(FAST,ORB),(BRISK,ORB),(HARRIS,ORB),(SHI_TOMASI,ORB),(KAZE,ORB),(AKAZE,ORB),(MSER,ORB),(AGAST,ORB),(GFTT,ORB),(CENSUREE,ORB),(ASIFT,ORB),(SUSAN,ORB),
                          (KAZE, SIFT),(KAZE, SURF),(KAZE, ORB),(KAZE, BRISK),(KAZE, FREAK),(KAZE, BRIEF),(KAZE, ROOT_SIFT),
                          (SIFT,KAZE),(SURF,KAZE),(ORB, KAZE),(FAST,KAZE),(BRISK,KAZE),(HARRIS,KAZE),(SHI_TOMASI,KAZE),(AKAZE,KAZE),(MSER,KAZE),(AGAST,KAZE),(GFTT,KAZE),(CENSUREE,KAZE),(ASIFT,KAZE),(SUSAN,KAZE),
                          (AKAZE, SIFT),(AKAZE, SURF),(AKAZE, ORB),(AKAZE, BRISK),(AKAZE, FREAK),(AKAZE, KAZE),(AKAZE, BRIEF),(AKAZE, ROOT_SIFT),
                          (SIFT,AKAZE),(SURF,AKAZE),(ORB, AKAZE),(FAST,AKAZE),(BRISK,AKAZE),(HARRIS,AKAZE),(SHI_TOMASI,AKAZE),(MSER,AKAZE),(AGAST,AKAZE),(GFTT,AKAZE),(CENSUREE,AKAZE),(ASIFT,AKAZE),(SUSAN,AKAZE),
                          (SURF, ROOT_SIFT),(ORB, ROOT_SIFT),(FAST, ROOT_SIFT),(BRISK, ROOT_SIFT),(HARRIS, ROOT_SIFT),(SHI_TOMASI,ROOT_SIFT),(KAZE, ROOT_SIFT),(AKAZE, ROOT_SIFT),(MSER, ROOT_SIFT),(AGAST, ROOT_SIFT),(GFTT, ROOT_SIFT),(CENSUREE, ROOT_SIFT),(ASIFT, ROOT_SIFT),(SUSAN, ROOT_SIFT),
                          (ASIFT,SURF),(ASIFT,ORB),(ASIFT,BRISK),(ASIFT,FREAK),(ASIFT,KAZE),(ASIFT,AKAZE),(ASIFT,BRIEF),(ASIFT,ROOT_SIFT)
                          ]

# combinations_to_remove2 = [(SIFT,SIFT),(SIFT,SURF),(SIFT,BRISK),(SIFT,FREAK),(SIFT,BRIEF),(SIFT,ROOT_SIFT),
#                            (SURF,SIFT),(SURF,SURF),(SURF,BRISK),(SURF,FREAK),(SURF,BRIEF),
#                            (ORB,ORB),
#                            (FAST,SIFT),(FAST,SURF),(FAST,BRISK),(FAST,FREAK),(FAST,BRIEF),
#                            (BRISK,SIFT),(BRISK,SURF),(BRISK,BRISK),(BRISK,FREAK),(BRISK,BRIEF),
#                            (HARRIS,SIFT),(HARRIS,BRISK),(HARRIS,FREAK),(HARRIS,BRIEF),
#                            (SHI_TOMASI,SIFT),(SHI_TOMASI,BRISK),(SHI_TOMASI,FREAK),(SHI_TOMASI,BRIEF),
#                            (KAZE,KAZE),
#                            (AKAZE,AKAZE),
#                            (MSER,SIFT),(MSER,SURF),(MSER,BRISK),(MSER,FREAK),(MSER,BRIEF),
#                            (AGAST,SIFT),(AGAST,SURF),(AGAST,BRISK),(AGAST,FREAK),(AGAST,BRIEF),
#                            (GFTT,SIFT),(GFTT,SURF),(GFTT,BRISK),(GFTT,FREAK),(GFTT,BRIEF)]

# combinations_to_remove += combinations_to_remove2


combinations_to_remove = list(set(combinations_to_remove))

for el in combinations_to_remove:
    try:
        det_des_combinations.remove(el)
    except Exception as e:
        print('{} NOT IN DICT'.format(el))

for dett, dess in det_des_combinations:
    print('Computing combination {}:{}'.format(dett,dess))
    # Set Variables
    detector_name = dett
    descriptor_name = dess

    # Obtain keypoints
    try:
        createKeypoints(detector_name, descriptor_name, dataset_path,all_at_once_dict[detector_name])
        removeUncommonPoints(detector_name, descriptor_name, dataset_path)
    except Exception as e:
        print('ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR!')
        print(e)
        with open('failed_combinations.txt','a+') as file:
            file.write(detector_name + '_' + descriptor_name +'\n')
        failed_combinations.append(detector_name + '_' + descriptor_name)
        continue

    # Evaluations
    list_of_APs, list_of_APs_i, list_of_APs_v = patchVerification(detector_name,
                                                                  descriptor_name,
                                                                  n,
                                                                  dataset_path,
                                                                  nr_iterations)

    with open('pV' + FILE_EXTENSION,'a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_APs),
                                           str(list_of_APs_i),
                                           str(list_of_APs_v)) + '\n')

    list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v = imageMatching(detector_name,
                                                                 descriptor_name,
                                                                 n,
                                                                 dataset_path,
                                                                 nr_iterations)
    with open('iM' + FILE_EXTENSION,'a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_mAPs),
                                           str(list_of_mAPs_i),
                                           str(list_of_mAPs_v)) + '\n')

    list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v = patchRetrieval(detector_name,
                                                                  descriptor_name,
                                                                  n,
                                                                  dataset_path,
                                                                  nr_iterations)

    with open('pR' + FILE_EXTENSION,'a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_mAPs),
                                           str(list_of_mAPs_i),
                                           str(list_of_mAPs_v)) + '\n')
