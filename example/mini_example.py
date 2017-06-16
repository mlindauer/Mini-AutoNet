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

import keras
import numpy as np

from mini_autonet.autonet import AutoNet

# dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

an = AutoNet()
an.fit(X=labels, Y=one_hot_labels, epochs=100)

Y = an.predict(X=labels)