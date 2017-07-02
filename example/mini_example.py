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

import keras
import numpy as np

from mini_autonet.autonet import AutoNet

# dummy data
N_CLASSES = 10
data = np.random.random((1000, 100))
labels = np.random.randint(N_CLASSES, size=(1000, 1))

an = AutoNet(max_layers=5, n_classes=N_CLASSES)
an.fit(X_train=data, y_train=labels, X_valid=data, y_valid=labels, max_epochs=10)

#Y = an.predict(X=labels)