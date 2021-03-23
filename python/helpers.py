
import importlib.util as imp
import numpy
if imp.find_spec("cupy"): import cupy
import os

def create_index_splits(Y_Subject, Y_Sex, splits = 10, seed=None):
    """ this method subdivides the given labels into optimal groups

        for the subject prediction labels, it divides the indices into equally sized groups,
        each containing equally many samples of each person.

        for the gender prediction labels (gender is linked to subject obviously) the
        data is split into partitions where no subject can reoccur
    """

    assert splits >= 3, 'At least three splits required'

    #number of samples
    assert Y_Subject.shape[0] == Y_Sex.shape[0], 'Number of Subject and Gender sample labels differ: {} vs {}'.format(Y_Subject.shape, Y_Sex.shape)

    N, P = Y_Subject.shape
    _, I = Y_Sex.shape

    #create global permutation sequence
    Permutation = numpy.arange(N)

    if seed is not None: #reseed the random generator
        numpy.random.seed(seed)
    Permutation = numpy.random.permutation(Permutation)

    #permute label matrices. also return this thing!
    Y_Subject = Y_Subject[Permutation,...]
    Y_Sex = Y_Sex[Permutation,...]

    #initialize index lists
    SubjectIndexSplits = [None]*splits

    #1) create a split over subject labels first by iterating over all person labels and subdividing them as equally as possible.
    for i in range(P):
        pIndices = numpy.where(Y_Subject[:,i] == 1)[0]

        #compute an approx equally sized partitioning.
        partitioning = numpy.linspace(0, len(pIndices), splits+1, dtype=int)
        for si in range(splits):
            #make sure index lists exist
            if SubjectIndexSplits[si] is None:
                SubjectIndexSplits[si] = []

            #spread subject label across those index lists
            if si == splits-1:
                #the last group.
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:])
            else:
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:partitioning[si+1]])


    #2) create a split over sex labels, balancing sex as good as possible but by avoiding the same subject label in more than one bin.
    #for sex recognition, we want to avoid the model to learn gait criteria of subjects and classify by that bias.
    #first split into sex groups and use them as queues
    sexQueues = [numpy.where(Y_Sex[:, i] == 1)[0].tolist() for i in range(I)]
    SexIndexSplits = [None]*splits
    currentSplit = 0

    #alternatingly move through sex lists and place people into splits accordingly.
    #remove those people from the sex queues accordingly.
    while sum([len(iQ) for iQ in sexQueues]) > 0:
        #make sure the split is populated
        if SexIndexSplits[currentSplit] is None:
            SexIndexSplits[currentSplit] = []

        #for each sex get next person, if this sex is not yet exhausted.
        for iQ in sexQueues:
            if len(iQ) == 0:
                continue

            #process lists/subjects:
            #find out who the next person is. get all those entries.
            pindex = numpy.where(Y_Subject[iQ[0], :])[0]
            #get all the indices for that person.
            pIndices = numpy.where(Y_Subject[:, pindex])[0]

            #remove this person from its respective queue
            for p in pIndices:
                iQ.remove(p)

            #and add it to its split group
            SexIndexSplits[currentSplit].extend(pIndices)

        #move split position
        currentSplit = (currentSplit + 1) % splits

    #return the indices for the subject recognition training, the gender recognition training and the original permutation to be applied on the data.
    return SubjectIndexSplits, SexIndexSplits, Permutation


def convIOdims(D,F,S):
    #helper method for computing output dims of 2d convolutions when giving D as data shape, F as filter shape and S as stride
    #D, F and S are expected to be scalar values
    D = float(D)
    F = float(F)
    S = float(S)
    return (D-F)/S + 1


def ensure_dir_exists(path_to_dir):
    if not os.path.isdir(path_to_dir):
        print('Target directory {} does not exist. Creating.'.format(path_to_dir))
        os.makedirs(path_to_dir)
    # else:
    #    print('Target directory {} exists.'.format(path_to_dir))

def trim_empty_classes(Y):
    # expects an input array shaped Y x C. removes label columns for classes without samples.
    n_per_col = Y.sum(axis=0)
    empty_cols = n_per_col == 0
    if numpy.any(empty_cols):
        print('{} Empty columns detected in label matrix shaped {}. Columns are: {}. Removing.'.format(empty_cols.sum(), Y.shape, numpy.where(empty_cols)[0]))
        Y = Y[:,~empty_cols]
        print('    shape is {} post column removal.'.format(Y.shape))
        return Y
    else:
        print('No empty columns detected in label matrix shaped {}'.format(Y.shape))
        return Y

def arrays_to_cupy(*args):
    assert imp.find_spec("cupy"), "module cupy not found/installed."
    return tuple([cupy.array(a) for a in args])

def arrays_to_numpy(*args):
    if not imp.find_spec("cupy"): #cupy has not been installed and imported -> arrays should be numpy
        return args
    else:
        return tuple([cupy.asnumpy(a) for a in args])


def force_device(model, arrays, device=None):
    #enforces the use of a specific device (cpu or gpu) for given models or arrays
    #converts the model in-place
    #returns the transferred arrays
    if device is None:
        return arrays
    elif isinstance(device, str) and device.lower() == 'none':
        return arrays
    elif isinstance(device, str) and device.lower() == 'cpu':
        print('Forcing model and associated arrays to CPU')
        model.to_cpu()
        return arrays_to_numpy(*arrays)
    elif isinstance(device, str) and device.lower() == 'gpu':
        print('Forcing model and associated arrays to GPU')
        assert imp.find_spec("cupy") is not None, "Model can not be forced to execute on GPU device. No GPU device present"
        model.to_gpu()
        return arrays_to_cupy(*arrays)
    else:
        raise ValueError("Unsure how to interpret input value '{}' in helpers.force_device".format(device))

def get_channel_wise_bounds(array):
    """
    Returns the channel-wise lower and upper bounds of some data, assuming a shape of
    (batchsize, [feature dims])

    Parameters:
    -----------
    array - numpy or cupy array of floats shaped (N, ...)


    Returns:
    --------
    tuple of arrays (lower, upper), each shaped (1, ...)
        the returned arrays are in CPU-accessible memory (ie numpy type arrays)
    """
    array = arrays_to_numpy(array)[0]
    lower = numpy.amin(array, axis=0, keepdims=True)
    upper = numpy.amax(array, axis=0, keepdims=True)
    return (lower, upper)

def l1loss(y_test, y_pred):
    return numpy.abs(y_pred - y_test).sum()/y_test.shape[0]

def accuracy(y_test, y_pred):
    y_test = numpy.argmax(y_test, axis=1)
    y_pred = numpy.argmax(y_pred, axis=1)
    return numpy.mean(y_test == y_pred)
