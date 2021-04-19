
#used for debugging and showcasing. can be removed at a later time.

import sys
print(sys.argv)
# data inputs are given in order test val train (3 datasets) or test val/train (2 datasets). If a single dataset is given, splits are generated as per parameterization
sys.argv += ['-d', './data/Phase-1_AIST_atMM_8.mat ./data/Phase-1_GaitRec_atMM_8.mat ./data/Phase-1_Gutenberg_atMM_8.mat',
            #'-d', './data/Phase-1_AIST_atMM_8.mat',
             '-o', './testing/blabla.mat',
             '-s', '3', #should be ignored with multiple data entries given
             '-a', 'SvmLinearL2C1e0',
             '-me', 'evaluate',
             #'-dt', None,
             '-dt', '1'
             ]

print(sys.argv)

#start debugging right here:
import gait_experiments