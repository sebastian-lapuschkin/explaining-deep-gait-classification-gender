
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@version: 1.0
@copyright: Copyright (c)  2019, Sebastian Lapuschkin
@license : BSD-2-Clause
'''

import argparse
import datetime
import os
import sys

import numpy
import numpy as numpy # no cupy import here, stay on the CPU in the main script.

import scipy.io as scio # scientific python package, which supports mat-file IO within python
import helpers
import eval_score_logs
import datetime
from termcolor import cprint, colored

import model
from model import *
from model.base import ModelArchitecture, ModelTraining
import train_test_cycle         #import main loop

current_datetime = datetime.datetime.now()
#setting up an argument parser for controllale command line calls
import argparse
parser = argparse.ArgumentParser(description="Train and evaluate Models on human gait recordings!")
parser.add_argument('-d',  '--data_path', type=str, nargs='*', help='Sets the path(s) to the dataset mat-file(s) to be processed: One (1) path will cause the script to create --splits splits from the data, creating training, test and validation sets. Two (2) paths will interpret the data as dedicated/prepared and test and training data, in that order. Three (3) paths will lead to the loading of these paths as dedicated test and validation and training sets, in that order.')
parser.add_argument('-o', '--output_dir', type=str, default='./output', help='Sets the output directory root for models and results. Default: "./output"')
parser.add_argument('-me', '--model_exists', type=str, default='evaluate', help='Sets the behavior of the code in case a model file has been found at the output location. "skip" (default) skips remaining execution loop and does nothing. "retrain" trains the model anew. "evaluate" only evaluates the model with test data')
parser.add_argument('-rs', '--random_seed', type=int, default=1234, help='Sets a random seed for the random number generator. Default: 1234')
parser.add_argument('-s', '--splits', type=int, default=10, help='The number of splits to divide the data into. Default: 10. Ignored if multiple datasets are given via -d or --data_path')
parser.add_argument('-a', '--architecture', type=str, default='SvmLinearL2C1e0', help='The name of the model architecture to use/train/evaluate. Can be any joint-specialization of model.base.ModelArchitecture and model.base.ModelTraining. Default: SvmLinearL2C1e0 ')
parser.add_argument('-tp', '--training_programme', type=str, default=None, help='The training regime for the (NN) model to follow. Can be any class from model.training or any class implementing model.base.ModelTraining. The default value None executes the training specified for the NN model as part of the class definition.')
parser.add_argument('-dn', '--data_name', type=str, default='GRF_AV', help='The feature name of the data behind --data_path to be processed. Default: GRF_AV')
parser.add_argument('-tn', '--target_name', type=str, default='Sex', help='The target type of the data behind --data_path to be processed. Default: Sex')
parser.add_argument('-sd', '--save_data', type=bool, default=True, help='Whether to save the training and split data at the output directory root or not. Default: True')
parser.add_argument('-ft', '--force_training_device', type=str, default=None, help='Force training to be performed on a specific device, despite the default chosen numeric backend? Options: cpu, gpu, None. Default: None: Pick as defined in model definition.')
parser.add_argument('-fe', '--force_evaluation_device', type=str, default=None, help='Force evaluat to be performed on a specific device, despite the default chosen numeric backend? Options: cpu, gpu, None. Default: None. NOTE: Execution on GPU is beneficial in almost all cases, due to the massive across-batch-parallelism.')
parser.add_argument('-rc', '--record_call', type=bool, default=False, help='Whether to record the current call to this script in an ouput file specified via -rf or --record-file. Default: False. Only records in case the script terminates gracefully')
parser.add_argument('-rf', '--record_file', type=str, default='./command_history.txt', help='Determines the file name into which the current call to this script is recorded')
ARGS = parser.parse_args()


################################
#           "Main"
################################

if ARGS.data_path is None:
    cprint(colored('No input data specified. Use -d/--data_path parameter. Exiting'), 'yellow')
    exit()
else:
    print('Whitespace-separating input data paths...')
    data_paths = ARGS.data_path[0].split()
    if len(data_paths) == 1:
        cprint(colored('One (1) input data path recognized: "{}". Generating data splits as per -s/--splits and -rs/--random_seed parameters.'.format(data_paths[0]), 'yellow'))

        #load matlab data as dictionary using scipy
        gaitdata = scio.loadmat(data_paths[0])

        # Feature -> Bodenreaktionskraft
        X_GRF_AV = gaitdata['Feature']
        Label_GRF_AV = gaitdata['Feature_GRF_AV_Label'][0][0]   # x 6 channel label
        cprint(colored('Sample count is : {}'.format(X_GRF_AV.shape[0]), 'yellow'))

        #transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
        X_GRF_AV = numpy.transpose(X_GRF_AV, [0, 2, 1])         # N x T x C

        # Targets -> Subject labels und gender labels
        Y_Subject = gaitdata['Target_Subject']                  # N x L, binary labels
        Y_Sex = gaitdata['Target_Sex']                          # N x 1 , binary labels

        #split data for experiments.
        Y_Sex_trimmed = helpers.trim_empty_classes(Y_Sex)
        Y_Subject_trimmed = helpers.trim_empty_classes(Y_Subject)
        SubjectIndexSplits, SexIndexSplits, Permutation = helpers.create_index_splits(Y_Subject_trimmed, Y_Sex_trimmed, splits=ARGS.splits, seed=ARGS.random_seed)

        #apply the permutation to the given data for the inputs and labels to match the splits again
        X_GRF_AV = X_GRF_AV[Permutation, ...]
        Y_Sex_trimmed = Y_Sex_trimmed[Permutation, ...]
        Y_Subject_trimmed = Y_Subject_trimmed[Permutation, ...]
        do_xval = True # normal "split data automatically" behavior


    elif len(data_paths) == 2:
        cprint(colored('Two (2) input data paths recognized. Using as follows: Test: "{}" , Val/Train: "{}". Ignoring -s/--splits and -rs/--random_seed parameters.'.format(*data_paths), 'yellow'))
        if not ARGS.target_name == 'Sex':
            cprint(colored('Warning! Prediction target "{}" incompatible with pre-computed data splits. Forcing prediction target "Sex"'.format(ARGS.target_name), 'yellow'))
            ARGS.target_name = 'Sex'

        #load matlab data as dictionary using scipy: 0 = test, 1 = val/train
        gaitdata = [scio.loadmat(data_paths[i]) for i in [0,1,1]]
        data_sizes = [g['Feature'].shape[0] for g in gaitdata]
        cprint(colored('Sample count of Test / Val / Train is :  {} / {} / {}'.format(*data_sizes), 'yellow'))

        # Feature -> Bodenreaktionskraft
        X_GRF_AV = numpy.concatenate([g['Feature'] for g in gaitdata], axis=0)
        Label_GRF_AV = gaitdata[0]['Feature_GRF_AV_Label'][0][0]   # x 6 channel label. Assume they are identical for all input data

        #transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
        X_GRF_AV = numpy.transpose(X_GRF_AV, [0, 2, 1])         # N x T x C

        # Targets -> Subject labels und gender labels
        Y_Sex = numpy.concatenate([g['Target_Sex'] for g in gaitdata], axis=0)                   # N x 1 , binary labels

        #split data for experiments.
        Y_Sex_trimmed = Y_Sex # we have to assume that the data is clean.
        Permutation = numpy.arange(Y_Sex_trimmed.shape[0])

        SexIndexSplits = []
        i_start = 0
        for d_size in data_sizes:
            SexIndexSplits.append([i for i in range(i_start, i_start + d_size)])
            i_start += d_size

        SubjectIndexSplits = [] #dummy
        Y_Subject_trimmed = []  #dummy
        Y_Subject = numpy.zeros((Y_Sex.shape[0],)) #dummy
        do_xval = False # no xval with precomputed and dedicated data splits


    elif len(data_paths) >= 3:
        cprint(colored('Three or more (3+) input data paths recognized. Using (the first three) as follows: Test: "{}" , Val: "{}" ,  Train: "{}". Ignoring -s/--splits and -rs/--random_seed parameters.'.format(*data_paths), 'yellow'))
        if not ARGS.target_name == 'Sex':
            cprint(colored('Warning! Prediction target "{}" incompatible with pre-computed data splits. Forcing prediction target "Sex"'.format(ARGS.target_name), 'yellow'))
            ARGS.target_name = 'Sex'

        #load matlab data as dictionary using scipy
        gaitdata = [scio.loadmat(data_paths[i]) for i in range(3)]
        data_sizes = [g['Feature'].shape[0] for g in gaitdata]
        cprint(colored('Sample count of Test / Val / Train is :  {} / {} / {}'.format(*data_sizes), 'yellow'))

        # Feature -> Bodenreaktionskraft
        X_GRF_AV = numpy.concatenate([g['Feature'] for g in gaitdata], axis=0)
        Label_GRF_AV = gaitdata[0]['Feature_GRF_AV_Label'][0][0]   # x 6 channel label. Assume they are identical for all input data

        #transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
        X_GRF_AV = numpy.transpose(X_GRF_AV, [0, 2, 1])         # N x T x C

        # Targets -> Subject labels und gender labels
        Y_Sex = numpy.concatenate([g['Target_Sex'] for g in gaitdata], axis=0)                   # N x 1 , binary labels

        #split data for experiments.
        Y_Sex_trimmed = Y_Sex # we have to assume that the data is clean.
        Permutation = numpy.arange(Y_Sex_trimmed.shape[0])

        SexIndexSplits = []
        i_start = 0
        for d_size in data_sizes:
            SexIndexSplits.append([i for i in range(i_start, i_start + d_size)])
            i_start += d_size

        SubjectIndexSplits = [] #dummy
        Y_Subject_trimmed = []  #dummy
        Y_Subject = numpy.zeros((Y_Sex.shape[0],)) #dummy
        do_xval = False # no xval with precomputed and dedicated data splits



arch = ARGS.architecture
if isinstance(arch, ModelArchitecture) and isinstance(arch, ModelTraining):
    pass # already a valid class
elif isinstance(arch,str):
    #try to get class from string name
    arch = model.get_architecture(arch)
else:
    raise ValueError('Invalid command line argument type {} for "architecture'.format(type(arch)))



training_regime =  ARGS.training_programme
if training_regime is None or isinstance(training_regime, ModelTraining):
    pass #default training behavior of the architecture class, or training class
elif isinstance(training_regime, str):
    if training_regime.lower() == 'none':
        training_regime = None #default training behavior of the architecture class, or training class
    else:
        training_regime = model.training.get_training(training_regime)
    #try to get class from string name



#register and then select available features
X, X_channel_labels = {'GRF_AV': (X_GRF_AV, Label_GRF_AV)}[ARGS.data_name]

#register and then select available targets
Y, Y_splits = {'Sex': (Y_Sex_trimmed, SexIndexSplits) , 'Subject': (Y_Subject_trimmed, SubjectIndexSplits)}[ARGS.target_name]

# this load of parameters could also be packed into a dict and thenn passed as **param_dict, if this were to be automated further.
train_test_cycle.run_train_test_cycle(
        X=X,
        Y=Y,
        L=X_channel_labels,
        LS=Y_Subject,
        S=Y_splits,
        P=Permutation,
        model_class=arch,
        output_root_dir=ARGS.output_dir,
        data_name=ARGS.data_name,
        target_name=ARGS.target_name,
        save_data_in_output_dir=ARGS.save_data,
        training_programme=training_regime, # model training behavior can be exchanged (for NNs), eg by using NeuralNetworkTrainingQuickTest instead of None. define new behaviors in model.training.py!
        do_this_if_model_exists=ARGS.model_exists,
        force_device_for_training=ARGS.force_training_device,
        force_device_for_evaluation=ARGS.force_evaluation_device, # computing heatmaps on gpu is always worth it for any model. requires a gpu, obviously
        do_xval=do_xval # True if one blob of data is given. False if dedicated splits have been loaded
)
eval_score_logs.run(ARGS.output_dir)

#record function call and parameters if we arrived here

if ARGS.record_call:
    print('Recording current call configuration to {}'.format(ARGS.record_file))
    helpers.ensure_dir_exists(os.path.dirname(ARGS.record_file))
    with open(ARGS.record_file, 'a') as f:
        argline = ' '.join(['--{} {}'.format(a, getattr(ARGS,a)) for a in vars(ARGS)])
        line = '{} : python {} {}'.format(current_datetime,
                                       sys.modules[__name__].__file__,
                                       argline)
        f.write('{}\n\n'.format(line))
