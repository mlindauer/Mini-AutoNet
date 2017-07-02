import random
import functools
import logging

import numpy as np
import matplotlib.pyplot as plt

from param_net.param_fcnet import ParamFCNetClassification

from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

class AutoNet(object):
    
    def __init__(self, max_layers:int=5, n_classes:int=2 ):
        self.logger = logging.getLogger("AutoNet")
        
        self.max_layers = max_layers
        self.n_classes = n_classes
    

    def fit(self, X_train, y_train, X_valid, y_valid, max_expochs:int):


        def obj_func(config): 
            pc = ParamFCNetClassification(config=config, n_feat=X_train.shape[1],
                                          n_classes=self.n_classes)
            history = pc.train(X_train=X_train, y_train=y_train, X_valid=X_valid,
                               y_valid=y_valid, n_epochs=max_expochs)
            final_loss = history.history["loss"][-1] 
            
            return final_loss


        taf = ExecuteTAFuncDict(obj_func)
        cs = ParamFCNetClassification.get_config_space(max_num_layers=self.max_layers)
        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": 42,
                                "cost_for_crash": 10, 
                                "cs": cs,
                                "deterministic": "true",
                                "output-dir": ""
                                })
        
        smac = SMAC(scenario=ac_scenario, 
                    tae_runner=taf,
                    rng=np.random.RandomState(42))
        incumbent = smac.optimize()