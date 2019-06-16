
# PIPELINE 3 JE ZA POKRENUTI SAMO JEDAN det+des
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

from utils import *
from evaluations import patchVerification, imageMatching, patchRetrieval
from visualizations import taskEvaluation
from const import *


project_root = '/home/kristijan/PhD/hpatches-benchmark/python/ISPA'
dataset_path = '/home/kristijan/PhD/hpatches-sequences-release/*'
n = 100
nr_iterations = 5
pV_results = {}
iM_results = {}
pR_results = {}
FILE_EXTENSION = '_LFNET_NEW.txt'


detector_name = LFNET
descriptor_name = LFNET

#createKeypoints(detector_name, descriptor_name, dataset_path)
#removeUncommonPoints(detector_name, descriptor_name, dataset_path)

# Evaluations
list_of_APs, list_of_APs_i, list_of_APs_v = patchVerification(detector_name,
                                                              descriptor_name,
                                                              n,
                                                              dataset_path,
                                                              nr_iterations)

with open('pV' + FILE_EXTENSION, 'a+') as file:
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
with open('iM' + FILE_EXTENSION, 'a+') as file:
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

with open('pR' + FILE_EXTENSION, 'a+') as file:
    file.write('{}_{}|{}|{}|{}'.format(detector_name,
                                       descriptor_name,
                                       str(list_of_mAPs),
                                       str(list_of_mAPs_i),
                                       str(list_of_mAPs_v)) + '\n')
