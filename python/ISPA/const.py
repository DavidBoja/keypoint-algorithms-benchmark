##############
# ALGORITHMS #
##############
SIFT = 'sift'
SURF = 'surf'
BRIEF = 'brief'
BRISK = 'brisk'
ORB = 'orb'
FAST = 'fast'
HARRIS = 'harris'
SHI_TOMASI = 'shi_tomasi'
KAZE = 'kaze'
AKAZE = 'akaze'

# descriptor-only
FREAK = 'freak'

# Det+desc template
ALGO_TEMPLATE = '{}_{}'

####################
# EVALUATION TASKS #
####################

VERIFICATION = 'Verification'
MATCHING = 'Matching'
RETRIEVAL = 'Retrieval'

TASKS = [VERIFICATION, MATCHING, RETRIEVAL]

###########################
# ALGORITHM DICTIONARIES #
##########################
import cv2
from own_implementations import *

all_detectors = {'sift':cv2.xfeatures2d.SIFT_create,
                 'surf':cv2.xfeatures2d.SURF_create,
                 'orb':cv2.ORB_create,
                 'fast':cv2.FastFeatureDetector_create,
                 'brisk':cv2.BRISK_create,
                 'harris':HarrisMataHarris,
                 'shi_tomasi':ShiTomasi,
                 'kaze':cv2.KAZE_create}

all_descriptors = {'sift':cv2.xfeatures2d.SIFT_create,
                   'surf':cv2.xfeatures2d.SURF_create,
                   'orb':cv2.ORB_create,
                   'brisk':cv2.BRISK_create,
                   'freak':cv2.xfeatures2d.FREAK_create,
                   'kaze':cv2.KAZE_create}

descriptor_distance = {'sift':cv2.NORM_L2,
                       'surf':cv2.NORM_L2,
                       'orb':cv2.NORM_HAMMING2,
                       'brisk':cv2.NORM_HAMMING2,
                       'freak':cv2.NORM_HAMMING2,
                       'kaze':cv2.NORM_L2,
                       # deep
                       'lfnet':cv2.NORM_L2,
                       'superpoint':cv2.NORM_L2}

#########
# PATHS #
#########

RESULTS_DIR = 'results/'
