import random
import logging

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition, AndConjunction, GreaterThanCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from smac.facade.smac_facade import SMAC

class AutoNet(object):
    
    def __init__(self, max_layers:int=5):
        self.model = None
        self._build_cs(max_layers=max_layers)
        self.logger = logging.getLogger("AutoNet")
    
    def _build_cs(self, max_layers:int=5):
        
        cs = ConfigurationSpace()
        self.cs = cs
        
        def _add(param, parents:list=None):
            cs.add_hyperparameter(param)
            
            if parents:
                conds = []
                for [parent,value] in parents:
                    cond = InCondition(
                        child=param, parent=parent, values=[value])
                    conds.append(cond)
                if len(conds) > 1:
                    cs.add_condition(AndConjunction(*conds))
                else:
                    cs.add_condition(conds[0])
                
        layers = UniformIntegerHyperparameter(
                name="num_layers", lower=1, upper=max_layers, default=3, log=True)
        _add(layers)

        for i in range(1,max_layers+1):
            layer_neurons = UniformIntegerHyperparameter(
                name="layer%d_neurons" %(i), lower=10, upper=400, default=128, log=True)
            _add(layer_neurons)
            act_func = CategoricalHyperparameter(
                name="layer%d_act_func" %(i), choices=["elu","relu","tanh","sigmoid"], default="relu")
            _add(act_func)
            dropout = UniformFloatHyperparameter(
                name="layer%d_dropout_rate" %(i), lower=0, upper=0.99, default=0.0)
            _add(dropout)
            if i > 1:
                gtc = GreaterThanCondition(child=layer_neurons, parent=layers, value=i-1)
                cs.add_condition(gtc)
                gtc = GreaterThanCondition(child=act_func, parent=layers, value=i-1)
                cs.add_condition(gtc)
                gtc = GreaterThanCondition(child=dropout, parent=layers, value=i-1)
                cs.add_condition(gtc)

        #batch size
        batch_size = UniformIntegerHyperparameter(
            name="batch_size", lower=1, upper=128, default=32, log=True)
        _add(batch_size)

        ## Optimizer
        optimizer = CategoricalHyperparameter(
            name="optimizer", choices=["SGD", "Adam"], default="Adam")
        _add(optimizer)
        
        ##SGD
        lr = UniformFloatHyperparameter(
            name="sgd:lr", lower=0.000001, upper=0.1, default=0.01, log=True)
        _add(lr, [[optimizer,"SGD"]])
        momentum = UniformFloatHyperparameter(
            name="sgd:momentum", lower=0.6, upper=0.999, default=0.9, log=True)
        _add(momentum, [[optimizer,"SGD"]])
            
        ## Adam
        lr = UniformFloatHyperparameter(
            name="adam:lr", lower=0.000001, upper=1.0, default=0.0001, log=True)
        _add(lr, [[optimizer,"Adam"]])
        beta_1 = UniformFloatHyperparameter(
            name="adam:beta_1", lower=0.7, upper=0.999999, default=0.9)
        _add(beta_1, [[optimizer,"Adam"]])
        beta_2 = UniformFloatHyperparameter(
            name="adam:beta_2", lower=0.9, upper=0.999999, default=0.999)
        _add(beta_2, [[optimizer,"Adam"]])
        epsilon = UniformFloatHyperparameter(
            name="adam:epsilon", lower=1e-20, upper=0.1, default=1e-08, log=True)
        _add(epsilon, [[optimizer,"Adam"]])
        decay = UniformFloatHyperparameter(
            name="adam:decay", lower=0.0, upper=0.1, default=0.000)
        _add(decay, [[optimizer,"Adam"]])
    
    def fit(self, X, Y, 
            epochs:int=1, 
            output_activation:str="softmax", 
            loss_func:str='categorical_crossentropy', 
            func_budget:int=1,
            do_plot:bool=True):
        
        if func_budget==1:
            config = self.cs.get_default_configuration()
        else:
            #TODO: run SMAC 
            pass
        
        self.logger.info("Final Configuration")
        self.logger.info(str(config))
        
        self.model = self._build_dnn(config=config,
                        n_input_neurons=X.shape[1],
                        n_output_neurons=Y.shape[1],
                        output_activation=output_activation)
        optimizer = self._build_optimizer(config=config)
        

        self.model.compile(
              loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        # es = EarlyStopping(monitor="val_loss", patience=1)
        
        history = self.model.fit(X, Y, epochs=epochs, batch_size=config["batch_size"],
                                 #validation_split=0.33,
                                 #callbacks=[es]
                                 )
        if do_plot is True:
            self._plot(history=history)
    
    def _build_dnn(self, 
                   config:Configuration,
                   n_input_neurons:int, 
                   n_output_neurons:int=1,
                   output_activation:str="softmax"):
        
        model = Sequential()
        model.add(Dense(units=config["layer1_neurons"], 
                             activation=config["layer1_act_func"],
                             input_dim=n_input_neurons))
        model.add(Dropout(rate=config["layer1_dropout_rate"]))
        for i in range(2,config["num_layers"]+1):
            model.add(Dense(units=config["layer%d_neurons" %(i)], activation=config["layer%d_act_func" %(i)]))
            model.add(Dropout(rate=config["layer%d_dropout_rate" %(i)]))
            
        # output layer
        model.add(Dense(units=n_output_neurons))
        model.add(Activation(output_activation))
        
        return model
        
    def _build_optimizer(self,
                         config:Configuration):
        
        if config["optimizer"] == "SGD":
            optimizer = keras.optimizers.SGD(lr=config["sgd:lr"], momentum=config["sgd:momentum"], nesterov=True)
        elif config["optimizer"] == "Adam":
            optimizer = keras.optimizers.Adam(lr=config["adam:lr"], 
                                          beta_1=config["adam:beta_1"], 
                                          beta_2=config["adam:beta_2"],
                                          epsilon=config["adam:epsilon"], 
                                          decay=config["adam:decay"]
                                          )
        return optimizer
    
    def _plot(self, history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["acc"])
        # plt.plot(history.history["val_acc"])
        plt.xlabel('epoch')
        plt.legend(['train loss', 'train acc'], loc='upper right')
        plt.savefig("loss_dnn_%d.png" %(random.randint(1,2**31)))
        
    def predict(self, X):
        
        if self.model is None:
            raise ValueError("Model was not fitted; please call first fit()")
        
        return self.model.predict(X)
        