##############
# ALGORITHMS #
##############
SIFT = 'sift'
ASIFT = 'asift'
SURF = 'surf'
BRIEF = 'brief'
BRISK = 'brisk'
ORB = 'orb'
FAST = 'fast'
HARRIS = 'harris'
SHI_TOMASI = 'shi_tomasi'
KAZE = 'kaze'
AKAZE = 'akaze'
MSER = 'mser'
AGAST = 'agast'
GFTT = 'gftt'
CENSUREE = 'censure'
ROOT_SIFT = 'root_sift'
SUSAN = 'susan'

# descriptor-only
FREAK = 'freak'

# deep
LFNET = 'lfnet'
SUPERPOINT = 'superpoint'
D2NET = 'd2net'

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

all_detectors = {'sift':cv2.xfeatures2d.SIFT_create,
                 'surf':cv2.xfeatures2d.SURF_create,
                 'orb':cv2.ORB_create,
                 'fast':cv2.FastFeatureDetector_create,
                 'brisk':cv2.BRISK_create,
                 #'harris':HarrisMataHarris,
                 #'shi_tomasi':ShiTomasi,
                 'kaze':cv2.KAZE_create,
                 'akaze':cv2.AKAZE_create,
                 'mser':cv2.MSER_create,
                 'agast':cv2.AgastFeatureDetector_create,
                 'gftt':cv2.GFTTDetector_create,
                 #'censure':CensureClass,
                 #'asift':ASIFTClass
                 #'susan':SusanClass
                 }

all_descriptors = {'sift':cv2.xfeatures2d.SIFT_create,
                   'surf':cv2.xfeatures2d.SURF_create,
                   'orb':cv2.ORB_create,
                   'brisk':cv2.BRISK_create,
                   'freak':cv2.xfeatures2d.FREAK_create,
                   'kaze':cv2.KAZE_create,
                   'akaze':cv2.AKAZE_create,
                   'brief':cv2.xfeatures2d.BriefDescriptorExtractor_create,
                   #'root_sift':RootSIFT
                   }

descriptor_distance = {'sift':cv2.NORM_L2,
                       'surf':cv2.NORM_L2,
                       'orb':cv2.NORM_HAMMING2,
                       'brisk':cv2.NORM_HAMMING2,
                       'freak':cv2.NORM_HAMMING2,
                       'kaze':cv2.NORM_L2,
                       'akaze':cv2.NORM_HAMMING2,

                       'brief':cv2.NORM_HAMMING2,
                       'root_sift':cv2.NORM_L2,
                       'lfnet':cv2.NORM_L2,
                       'superpoint':cv2.NORM_L2,
                       'd2net':cv2.NORM_L2}

all_at_once_dict = {'sift':False,
                    'surf':False,
                    'orb':False,
                    'fast':False,
                    'brisk':False,
                    'harris':False,
                    'shi_tomasi':False,
                    'kaze':False,
                    'akaze':False,
                    'mser':False,
                    'agast':False,
                    'gftt':False,
                    'censure':False,
                    'asift':True,
                    #'susan':False


                   # deep
                   'lfnet':True,
                   'superpoint':True}

#########
# PATHS #
#########

RESULTS_DIR = 'results/'
