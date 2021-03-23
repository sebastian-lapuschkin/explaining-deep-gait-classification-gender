# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:01:45 2021

@author: horst
"""

# Phase - 1
runfile('V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python\gait_experiments.py', args='-d ./data/Phase-1_Gutenberg_atMM -o ./output/Phase-1_Gutenberg_atMM.mat -s 10 -a CnnAshort', wdir='V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python')

runfile('V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python\gait_experiments.py', args='-d ./data/Phase-1_AIST_atMM -o ./output/Phase-1_AIST_atMM.mat -s 10 -a SvmLinearL2C1e0', wdir='V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python')	
runfile('V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python\gait_experiments.py', args='-d ./data/Phase-1_AIST_atMM -o ./output/Phase-1_AIST_atMM.mat -s 10 -a SvmLinearL2C1em1', wdir='V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python')	
runfile('V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python\gait_experiments.py', args='-d ./data/Phase-1_AIST_atMM -o ./output/Phase-1_AIST_atMM.mat -s 10 -a SvmLinearL2C1ep1', wdir='V:\Trainingswissenschaft\Horst\Gait - XAI - Gender\interpretable-deep-gait-injury\python')	
