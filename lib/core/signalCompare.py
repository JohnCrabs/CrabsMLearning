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

SC_EXEC_STATE = "Execution State"
SC_EXEC_FUNC = "Execution Function"

SC_PEARSON_R = "r"
SC_PEARSON_R2 = "R"
SC_PEARSON_P = "p"

# FIGURE FLAGS
SC_PEARSON_LABEL_Y_AXIS_SIGNAL_1 = 'Pearson Label Signal 1'
SC_PEARSON_LABEL_Y_AXIS_SIGNAL_2 = 'Pearson Label Signal 2'
SC_PEARSON_LABEL_TITLE = 'Pearson Label Title'
SC_PEARSON_LABEL_X_AXIS = 'Pearson Label Title'


class SignalCompare:
    def __init__(self):
        self._SC_dictMethods = {}

    def setSC_dict(self):
        self._SC_dictMethods[SC_PEARSON_CORRELATION] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_PearsonCorrelation,
            SC_PEARSON_R: np.nan,
            SC_PEARSON_R2: np.nan,
            SC_PEARSON_P: np.nan,
            SC_PEARSON_LABEL_Y_AXIS_SIGNAL_1: 'Signal 1',
            SC_PEARSON_LABEL_Y_AXIS_SIGNAL_2: 'Signal 1',
            SC_PEARSON_LABEL_TITLE: 'Pearson Correlation',
            SC_PEARSON_LABEL_X_AXIS: 'Frames'
        }

        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_TimeLaggedCrossCorrelation}

        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION_NO_SPLITS] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_TimeLaggedCrossCorrelationNoSplits}

        self._SC_dictMethods[SC_ROLLING_WINDOW_TIME_LAGGED_CROSS_CORRELATION] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_RollingWindowTimeLaggedCrossCorrelation}

        self._SC_dictMethods[SC_DYNAMIC_TIME_WARPING] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_DynamicTimeWarping}

    # ***************************** #
    # ***** SETTERS / GETTERS ***** #
    # ***************************** #

    def setPearsonCorr_state(self, state: bool):
        self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_EXEC_STATE] = state

    def setTimeLaggedCrossCorrelation_state(self, state: bool):
        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION][SC_EXEC_STATE] = state

    def setTimeLaggedCrossCorrelationNoSplits_state(self, state: bool):
        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION_NO_SPLITS][SC_EXEC_STATE] = state

    def setRollingWindowTimeLaggedCrossCorrelation_state(self, state: bool):
        self._SC_dictMethods[SC_ROLLING_WINDOW_TIME_LAGGED_CROSS_CORRELATION][SC_EXEC_STATE] = state

    def setDynamicTimeWarping_state(self, state: bool):
        self._SC_dictMethods[SC_DYNAMIC_TIME_WARPING][SC_EXEC_STATE] = state

    def getPearsonCorr_state(self):
        return self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_EXEC_STATE]

    def getTimeLaggedCrossCorrelation_state(self):
        return self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION][SC_EXEC_STATE]

    def getTimeLaggedCrossCorrelationNoSplits_state(self):
        return self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION_NO_SPLITS][SC_EXEC_STATE]

    def getRollingWindowTimeLaggedCrossCorrelation_state(self):
        return self._SC_dictMethods[SC_ROLLING_WINDOW_TIME_LAGGED_CROSS_CORRELATION][SC_EXEC_STATE]

    def getDynamicTimeWarping_state(self):
        return self._SC_dictMethods[SC_DYNAMIC_TIME_WARPING][SC_EXEC_STATE]

    def getPearson_r(self):
        return self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_R]

    def getPearson_R2(self):
        return self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_R2]

    def getPearson_P(self):
        return self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_P]

    # *************************** #
    # ***** METHODS EXECUTE ***** #
    # *************************** #

    def method_PearsonCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        pass

    def method_TimeLaggedCrossCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        pass

    def method_TimeLaggedCrossCorrelationNoSplits(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        pass

    def method_RollingWindowTimeLaggedCrossCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        pass

    def method_DynamicTimeWarping(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        pass

    def signComp_exec_(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str):
        for _method_ in self._SC_dictMethods.keys():
            if self._SC_dictMethods[_method_][SC_EXEC_STATE]:
                self._SC_dictMethods[_method_][SC_EXEC_FUNC](arrData1, arrData2, exportFigDirPath)