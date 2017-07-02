import logging
import inspect
import math
import time

import numpy as np

from smac.configspace import Configuration
from smac.tae.execute_ta_run import StatusType, ExecuteTARun
from smac.utils.constants import MAXINT


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class SimpleTAFunc(ExecuteTARun):
    """Baseclass to execute target algorithms which are Python functions.
    """

    def __init__(self, ta, stats=None, runhistory=None, run_obj="quality",
                 par_factor=1, cost_for_crash=float(MAXINT)):

        super().__init__(ta=ta, stats=stats, runhistory=runhistory,
                         run_obj=run_obj, par_factor=par_factor,
                         cost_for_crash=cost_for_crash)

        self.kwargs = None

    def start(self, config: Configuration,
              instance: str,
              cutoff: float=None,
              seed: int=12345,
              instance_specific: str="0",
              capped: bool=False,
              **kwargs):

        self.kwargs = kwargs

        return super().start(config,
                             instance,
                             cutoff,
                             seed,
                             instance_specific,
                             capped)

    def run(self, config, instance=None,
            cutoff=None,
            seed=12345,
            instance_specific="0"):
        """
            runs target algorithm <self.ta> with configuration <config> for at
            most <cutoff> seconds, allowing it to use at most <memory_limit>
            RAM.

            Whether the target algorithm is called with the <instance> and
            <seed> depends on the subclass implementing the actual call to
            the target algorithm.

            Parameters
            ----------
                config : dictionary (or similar)
                    dictionary param -> value
                instance : str
                    problem instance
                cutoff : int, optional
                    Wallclock time limit of the target algorithm. If no value is
                    provided no limit will be enforced.
                memory_limit : int, optional
                    Memory limit in MB enforced on the target algorithm If no
                    value is provided no limit will be enforced.
                seed : int
                    random seed
                instance_specific: str
                    instance specific information (e.g., domain file or solution)
            Returns
            -------
                status: enum of StatusType (int)
                    {SUCCESS, TIMEOUT, CRASHED, ABORT}
                cost: float
                    cost/regret/quality/runtime (float) (None, if not returned by TA)
                runtime: float
                    runtime (None if not returned by TA)
                additional_info: dict
                    all further additional run information
        """

        start_time = time.time()
        rval = self.ta(config, **self.kwargs)
        runtime = time.time() - start_time

        if isinstance(rval, tuple):
            result = rval[0]
            additional_run_info = rval[1]
        else:
            result = rval
            additional_run_info = {}

        status = StatusType.SUCCESS
        cost = result

        return status, cost, runtime, additional_run_info
