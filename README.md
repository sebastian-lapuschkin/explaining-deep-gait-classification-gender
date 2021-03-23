# This is a WIP progress repo
and effectively a fork of
[this repo](https://github.com/sebastian-lapuschkin/explaining-deep-gait-classification), with modifications from @fabian-horst

## Installation
In folder `python`, the file `install.sh` contains instructions to setup [`Miniconda3`](https://docs.conda.io/en/latest/miniconda.html)-based virtual environments for python, as required by our code.
Option A only considers CPU hardware, while option B enables GPU support
for neural network training and evaluation. Comment/uncomment the lines appropriately.

All recorded gait data used in the paper is available in folder `python/data`.
Training- and evaluation scripts for fully reproducing the data splits, models and prediction explanations are
provided with files `python/gait_experiments_batch*.py`.
The folder `sge` contains files `*.args`, presenting the mentioned training-evaluation runs as (probably more) handy command line parameters, one per line, either to be called directly as
```
python gait_experiments.py ${ARGS_LINE}
```
or to be submitted to a SUN Grid Engine with
```
python sge_job_simple.py your_file_of_choice.args
```
Some paths and variables need to be adjusted.


