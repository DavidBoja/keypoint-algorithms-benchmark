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
ALGO_TEMPLATE = '{}+{}'

####################
# EVALUATION TASKS #
####################

VERIFICATION = 'Verification'
MATCHING = 'Matching'
RETRIEVAL = 'Retrieval'

TASKS = [VERIFICATION, MATCHING, RETRIEVAL]
