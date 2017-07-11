##########################################################
# boiler plate to avoid installation
##########################################################
import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)
logging.basicConfig(level="INFO")
##########################################################

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from mini_autonet.autonet import AutoNet

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-X", required=True, help="X to be read with numpy.loadtxt()")
parser.add_argument("-Y", required=True, help="Y to be read with numpy.loadtxt() (not one-hot encoded)")
parser.add_argument("--X_valid", default=None, help="validation X to be read with numpy.loadtxt()")
parser.add_argument("--Y_valid", default=None, help="validation Y to be read with numpy.loadtxt() (not one-hot encoded)")
parser.add_argument("-C", required=True, type=int, help="number of classes")
parser.add_argument("-E", type=int, default=100, help="max epochs")
parser.add_argument("-R", type=int, default=100, help="runcount limit")
parser.add_argument("-L", type=int, default=10, help="maximum number of layers")

args_ = parser.parse_args()

X_train = np.loadtxt(args_.X)
Y_train = np.loadtxt(args_.Y)

if args_.X_valid:
    X_valid = np.loadtxt(args_.X_valid)
else:
    X_valid = X_train

if args_.Y_valid:
    Y_valid = np.loadtxt(args_.Y_valid)
else:
    Y_valid = Y_train
    

an = AutoNet(max_layers=args_.L, n_classes=args_.C)
an.fit(X_train=X_train, y_train=Y_train, X_valid=X_valid, y_valid=Y_valid, 
       max_epochs=args_.E, runcount_limit=args_.R)