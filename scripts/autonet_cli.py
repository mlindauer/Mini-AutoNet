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
logging.basicConfig(level="DEBUG")
##########################################################

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from mini_autonet.autonet import AutoNet

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("-X", help="X to be read with numpy.loadtxt()")
parser.add_argument("-Y", help="Y to be read with numpy.loadtxt() (not one-hot encoded)")
parser.add_argument("-C", type=int, help="number of classes")
parser.add_argument("-E", type=int, default=100, help="max epochs")
parser.add_argument("-R", type=int, default=100, help="runcount limit")
parser.add_argument("-L", type=int, default=10, help="maximum number of layers")

args_ = parser.parse_args()

X = np.loadtxt(args_.X)
Y = np.loadtxt(args_.Y)

an = AutoNet(max_layers=10, n_classes=args_.C)
an.fit(X_train=X, y_train=Y, X_valid=X, y_valid=Y, 
       max_epochs=args_.E, runcount_limit=args_.L)