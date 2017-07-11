import sys
import time
import copy
import logging
import typing
from collections import Counter
from collections import OrderedDict

import numpy as np

from smac.optimizer.objective import sum_cost
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT, MAX_CUTOFF
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, RunKey
from smac.tae.execute_ta_run import StatusType, BudgetExhaustedException, CappedRunException, ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

__author__ = " Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Intensifier(object):
    '''
     Races challengers against an incumbent (a.k.a. SMAC's intensification
     procedure).
    '''

    def __init__(self, tae_runner: ExecuteTARun, stats: Stats,
                 traj_logger: TrajLogger, rng: np.random.RandomState,
                 run_limit: int=MAXINT,
                 max_epochs: int=100):
        '''
        Constructor

        Parameters
        ----------
        tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
            target algorithm run executor
        stats: Stats
            stats object
        traj_logger: TrajLogger
            TrajLogger object to log all new incumbents
        rng : np.random.RandomState
        run_limit : int
            Maximum number of target algorithm runs per call to intensify.
        max_epochs : int
            maximum number of epochs (maximum calls to tae)
        '''
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.stats = stats
        self.traj_logger = traj_logger
            
        self.run_limit = run_limit
        self.rs = rng

        # scenario info
        self.tae_runner = tae_runner

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

        self._num_run = 0
        self._chall_indx = 0
        
        self.max_epochs = max_epochs
        
        self.learning_curves = []

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: Configuration,
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: int=MAXINT):
        '''
            running intensification to determine the incumbent configuration.
            Side effect: adds runs to run_history

            Implementation of Procedure 2 in Hutter et al. (2011).

            Parameters
            ----------
            challengers : typing.List[Configuration]
                promising configurations
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate error across instances
            time_bound : int, optional (default=2 ** 31 - 1)
                time in [sec] available to perform intensify

            Returns
            -------
            incumbent: Configuration()
                current (maybe new) incumbent configuration
            inc_perf: float
                empirical performance of incumbent configuration
        '''

        self.start_time = time.time()

        if time_bound < 0.01:
            raise ValueError("time_bound must be >= 0.01")

        self._num_run = 0
        self._chall_indx = 0

        # Line 1 + 2
        for challenger in challengers:
            if challenger == incumbent:
                self.logger.warning(
                    "Challenger was the same as the current incumbent; Skipping challenger")
                continue

            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug(
                    "Configuration origin: %s", challenger.origin)

            try:
                # Lines 8-17
                incumbent = self._race_challenger(challenger=challenger,
                                                  incumbent=incumbent,
                                                  run_history=run_history,
                                                  aggregate_func=aggregate_func)
                
            except BudgetExhaustedException:
                # We return incumbent, SMBO stops due to its own budget checks
                inc_perf = run_history.get_cost(incumbent)
                self.logger.debug("Budget exhausted; Return incumbent")
                return incumbent, inc_perf

            if self._chall_indx > 1 and self._num_run > self.run_limit:
                self.logger.debug(
                    "Maximum #runs for intensification reached")
                break
            elif self._chall_indx > 1 and time.time() - self.start_time - time_bound >= 0:
                self.logger.debug("Timelimit for intensification reached ("
                                  "used: %f sec, available: %f sec)" % (
                                      time.time() - self.start_time, time_bound))
                break

        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = aggregate_func(incumbent, run_history, inc_runs)
        self.logger.info("Updated estimated error of incumbent on %d runs: %.4f" % (
            len(inc_runs), inc_perf))

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return incumbent, inc_perf

    def _race_challenger(self, challenger: Configuration, 
                         incumbent: Configuration, 
                         run_history: RunHistory,
                         aggregate_func: typing.Callable):
        '''
            aggressively race challenger against incumbent

            Parameters
            ----------
            challenger : Configuration
                configuration which challenges incumbent
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate performance across instances

            Returns
            -------
            new_incumbent: Configuration
                either challenger or incumbent
        '''
        # at least one run of challenger
        # to increase chall_indx counter
        first_run = False
        inc_perf = run_history.get_cost(incumbent)

        learning_curve = []
        
        self._num_run += 1
        self._chall_indx += 1
        
        pc = None
        for epoch in range(self.max_epochs):
            status, cost, time, add_info = self.tae_runner.start(
                                            config=challenger,
                                            instance=None,
                                            seed=0,
                                            cutoff=2**32-1,
                                            instance_specific=None,
                                            pc=pc)
            pc = add_info["model"]
            learning_curve.append(cost)
            
        # delete model in runhistory to be more memory efficient
        chall_id = run_history.config_ids[challenger]
        runkey = RunKey(chall_id, None, 0)
        runvalue = run_history.data[runkey]
        del runvalue.additional_info["model"]
        
        if epoch == self.max_epochs -1:
            self.learning_curves.append(learning_curve)
        
        chal_perf = cost
        
        if cost < inc_perf:
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f)" % (
                chal_perf, inc_perf))
            # Show changes in the configuration
            params = sorted([(param, incumbent[param], challenger[param]) for param in
                             challenger.keys()])
            self.logger.info("Changes in incumbent:")
            for param in params:
                if param[1] != param[2]:
                    self.logger.info("  %s : %r -> %r" % (param))
                else:
                    self.logger.debug("  %s remains unchanged: %r" %
                                      (param[0], param[1]))
            incumbent = challenger
            self.stats.inc_changed += 1
            self.traj_logger.add_entry(train_perf=chal_perf,
                           incumbent_id=self.stats.inc_changed,
                           incumbent=challenger)
        else:
            self.logger.debug("Incumbent (%.4f) is better than challenger (%.4f)" % (inc_perf, chal_perf))

        return incumbent

