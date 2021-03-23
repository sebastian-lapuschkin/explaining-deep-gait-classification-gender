# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:01:45 2021

@author: horst
"""

# Phase - 1
#runfile('V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python\gait_experiments.py', args='-d ./data/Phase-2_GaitRec-Gutenberg-AIST_atMM -o ./output/Phase-2_GaitRec-Gutenberg-AIST_atMM.mat -s 3 -a CnnC3', wdir='V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python')
import sys
print(sys.argv)
sys.argv += ['-d', './data/Phase-2_GaitRec-Gutenberg-AIST_atMM',
             '-o', './output/Phase-2_GaitRec-Gutenberg-AIST_atMM.mat',
             #'-d', './data/Phase-1_AIST_atMM',
             '-s', '3',
             '-a', 'CnnC3']

print(sys.argv)

#start debugging right here:
import gait_experiments