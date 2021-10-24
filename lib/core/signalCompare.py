import os
import dtw

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

import matplotlib
import matplotlib.pyplot as plt

# *************************************************************************************************** #

matplotlib.use("Agg")  # Set matplotlib to use non-interface (don't plot figures)

SC_PEARSON_CORRELATION = 'Pearson Correlation'
SC_TIME_LAGGED_CROSS_CORRELATION = 'Time Lagged Cross Correlation'
SC_TIME_LAGGED_CROSS_CORRELATION_NO_SPLITS = 'Time Lagged Cross Correlation No Splits'
SC_ROLLING_WINDOW_TIME_LAGGED_CROSS_CORRELATION = 'Rolling Window Time Lagged Cross Correlation'
SC_DYNAMIC_TIME_WARPING = 'Dynamic Time Warping'

SC_METHODS_LIST = [SC_PEARSON_CORRELATION,
                   SC_TIME_LAGGED_CROSS_CORRELATION,
                   SC_TIME_LAGGED_CROSS_CORRELATION_NO_SPLITS,
                   SC_ROLLING_WINDOW_TIME_LAGGED_CROSS_CORRELATION,
                   SC_DYNAMIC_TIME_WARPING
                   ]


class SignalCompare:
    def __init__(self):
        self._SC_dictMethods = {}

    def setSC_dict(self):
        for _method_ in SC_METHODS_LIST:
            self._SC_dictMethods[_method_] = {}

    def method_PearsonCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray):
        pass