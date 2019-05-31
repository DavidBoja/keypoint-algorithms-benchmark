
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

from own_implementations import HarrisMataHarris, ShiTomasi
from utils import *
from evaluations import patchVerification, imageMatching, patchRetrieval
#from visualizations import taskEvaluation

project_root = '/home/davidboja/PycharmProjects/FER/hpatches-benchmark/python/ISPA'
dataset_path = project_root + '/hpatches-sequences-release/*'
n = 100
failed_combinations = []

det_des_combinations = list(itertools.product(list(all_detectors.keys()),
                                              list(all_descriptors.keys()) )
                                              )

combinations_to_remove = [(ORB, SIFT),
                          (ORB, SURF),
                          (ORB, BRISK),
                          (ORB, FREAK),
                          (ORB, KAZE),
                          (SIFT,ORB),
                          (SURF,ORB),
                          (FAST,ORB),
                          (BRISK,ORB),
                          (HARRIS,ORB),
                          (SHI_TOMASI,ORB),
                          (KAZE,ORB),
                          ]
# drugi_put_vrtim = [(FAST,SIFT),
#                     (FAST,BRISK),
#                     (SIFT,SIFT),
#                     (SIFT,SURF),
#                     (SIFT,BRISK),
#                     (SIFT,FREAK),
#                     (SURF,SIFT),
#                     (SURF,SURF)]

for el in combinations_to_remove:
    det_des_combinations.remove(el)

# for el in drugi_put_vrtim:
#     det_des_combinations.remove(el)


for dett, dess in det_des_combinations:
    print('Computing combination {}:{}'.format(dett,dess))
    # Set Variables
    detector_name = dett
    descriptor_name = dess

    # Obtain keypoints
    try:
        createKeypoints(detector_name, descriptor_name, dataset_path)
        removeUncommonPoints(detector_name, descriptor_name, dataset_path)
    except Exception as e:
        print('ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
        with open('failed_combinations.txt','a+') as file:
            file.write(detector_name + '_' + descriptor_name +'\n')
        failed_combinations.append(detector_name + '_' + descriptor_name)
        continue

    # Evaluations
    list_of_APs, list_of_APs_i, list_of_APs_v = patchVerification(detector_name,
                                                                  descriptor_name,
                                                                  n,
                                                                  dataset_path,
                                                                  5)

    with open('pV2.txt','a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_APs),
                                           str(list_of_APs_i),
                                           str(list_of_APs_v)) + '\n')

    list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v = imageMatching(detector_name,
                                                                 descriptor_name,
                                                                 n,
                                                                 dataset_path,
                                                                 5)
    with open('iM2.txt','a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_mAPs),
                                           str(list_of_mAPs_i),
                                           str(list_of_mAPs_v)) + '\n')

    list_of_mAPs, list_of_mAPs_i, list_of_mAPs_v = patchRetrieval(detector_name,
                                                                  descriptor_name,
                                                                  n,
                                                                  dataset_path,
                                                                  5)

    with open('pR2.txt','a+') as file:
        file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                           descriptor_name,
                                           str(list_of_mAPs),
                                           str(list_of_mAPs_i),
                                           str(list_of_mAPs_v)) + '\n')



    #deleteAllKeypoints(dataset_path)
