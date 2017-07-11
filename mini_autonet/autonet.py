import random
import functools
import logging

import numpy as np
import matplotlib.pyplot as plt

from param_net.param_fcnet import ParamFCNetClassification
from keras.losses import categorical_crossentropy
from keras import backend as K

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from mini_autonet.intensification.intensification import Intensifier
from mini_autonet.tae.simple_tae import SimpleTAFunc

class AutoNet(object):
    
    def __init__(self, max_layers:int=5, n_classes:int=2 ):
        self.logger = logging.getLogger("AutoNet")
        
        self.max_layers = max_layers
        self.n_classes = n_classes
    

    def fit(self, X_train, y_train, X_valid, y_valid, 
            max_epochs:int,
            runcount_limit:int=100,
            loss_func=categorical_crossentropy):


        def obj_func(config, instance=None, seed=None, pc=None):
            # continuing training if pc is given
            # otherwise, construct new DNN
            if pc is None:
                K.clear_session()
                pc = ParamFCNetClassification(config=config, n_feat=X_train.shape[1],
                                              n_classes=self.n_classes,
                                              loss_function=loss_func)
                
            history = pc.train(X_train=X_train, y_train=y_train, X_valid=X_valid,
                               y_valid=y_valid, n_epochs=1)
            final_loss = history.history["val_loss"][-1] 
            
            return final_loss, {"model": pc}


        taf = SimpleTAFunc(obj_func)
        cs = ParamFCNetClassification.get_config_space(max_num_layers=self.max_layers)
        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": max_epochs*runcount_limit,
                                "cost_for_crash": 10, 
                                "cs": cs,
                                "deterministic": "true",
                                "output-dir": ""
                                })
        
        intensifier = Intensifier(tae_runner=taf, stats=None,
                 traj_logger=None, 
                 rng=np.random.RandomState(42),
                 run_limit=100,
                 max_epochs=max_epochs)
        
        smac = SMAC(scenario=ac_scenario, 
                    tae_runner=taf,
                    rng=np.random.RandomState(42),
                    intensifier=intensifier)
        incumbent = smac.optimize()