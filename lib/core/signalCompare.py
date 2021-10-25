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
SC_PEARSON_LABEL_Y_AXIS_ROLLING_R = 'Pearson Rolling R'
SC_PEARSON_LABEL_TITLE = 'Pearson Label Title'
SC_PEARSON_LABEL_X_AXIS = 'Pearson Label X Axis'

SC_TIME_LAGGED_CROSS_CORRELATION_OFFSET = 'Offset'


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    wrap
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


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
            SC_PEARSON_LABEL_Y_AXIS_SIGNAL_1: 'Signal-1',
            SC_PEARSON_LABEL_Y_AXIS_SIGNAL_2: 'Signal-2',
            SC_PEARSON_LABEL_Y_AXIS_ROLLING_R: 'Signal-[1,2] - Rolling R',
            SC_PEARSON_LABEL_TITLE: 'Pearson Correlation',
            SC_PEARSON_LABEL_X_AXIS: 'Frames'
        }

        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION] = {
            SC_EXEC_STATE: False,
            SC_EXEC_FUNC: self.method_TimeLaggedCrossCorrelation,
            SC_TIME_LAGGED_CROSS_CORRELATION: np.nan
        }

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
    def method_PearsonCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        labelYaxis_1 = self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_LABEL_Y_AXIS_SIGNAL_1]
        labelYaxis_2 = self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_LABEL_Y_AXIS_SIGNAL_2]
        labelYaxisRollingR = self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_LABEL_Y_AXIS_ROLLING_R]
        xAxisLabel = self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_LABEL_X_AXIS]
        title = self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_LABEL_TITLE]

        r_window = int(0.25*arrData1.shape[0])
        print('rolling_window = ', r_window)
        if r_window < 3:
            r_window = 3

        df_arrData = pd.DataFrame({labelYaxis_1: arrData1, labelYaxis_2:arrData2})

        df_arrData = df_arrData.dropna()  # Remove na values if exist

        pearson_r, pearson_p = stats.pearsonr(df_arrData[labelYaxis_1], df_arrData[labelYaxis_2])
        pearson_R = pearson_r*pearson_r

        self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_R] = pearson_r
        self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_R2] = pearson_R
        self._SC_dictMethods[SC_PEARSON_CORRELATION][SC_PEARSON_P] = pearson_p

        # Export Figure
        f, ax = plt.subplots(figsize=(9.60, 6.40))
        color = 'tab:red'
        df_arrData[labelYaxis_1].rolling(window=r_window, center=True).median().plot(ax=ax, color=color)
        ax.set_xlabel(xAxisLabel)
        ax.set_ylabel(labelYaxis_1)
        ax.tick_params(axis='y', labelcolor=color)
        str_title = title + \
                    '\n' f"r = {np.round(pearson_r, 2)}, " + f"R = {np.round(pearson_R, 2)}, " + \
                    '\n' f"p = {np.round(pearson_p, 7)}"
        ax.set(title=str_title)
        ax.legend().remove()

        ax2 = ax.twinx()  # instantiate a second axis that shares the same x-axis
        color = 'tab:blue'
        df_arrData[labelYaxis_2].rolling(window=r_window, center=True).median().plot(ax=ax2, color=color)
        ax2.set_ylabel(labelYaxis_2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend().remove()

        f.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)
        f.tight_layout()
        final_export_path = os.path.normpath(exportFigDirPath) + '/' + 'FigNo00_' + exportFigFileName + \
                            '_PearsonRollingMedian.png'
        plt.savefig(final_export_path)
        plt.close()

        # Interpolate missing data.
        df_interpolated = df_arrData.interpolate()
        # Compute rolling window synchrony
        rolling_r = df_interpolated[labelYaxis_1].rolling(window=r_window, center=True).corr(df_interpolated[labelYaxis_2])
        _, ax = plt.subplots(2, 1, figsize=(14, 6))
        df_arrData.rolling(window=r_window, center=True).median().plot(ax=ax[0])
        ax[0].set(xlabel=xAxisLabel, ylabel=labelYaxisRollingR)
        ax[0].legend().remove()
        rolling_r.plot(ax=ax[1])
        ax[1].set(xlabel=xAxisLabel, ylabel='Pearson r')
        ax[1].legend().remove()
        plt.suptitle("Correlation Data and Rolling Window Correlation (window size = " + str(r_window) + ' )')
        plt.tight_layout()
        final_export_path = os.path.normpath(exportFigDirPath) + '/' + 'FigNo01_' + exportFigFileName + \
                            '_PearsonRolling_R.png'
        plt.savefig(final_export_path)
        plt.close()

    def method_TimeLaggedCrossCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        d1 = arrData1
        d2 = arrData2

        timeStep_mul_FPS = int(0.2 * arrData1.shape[0])
        rs = [crosscorr(d1, d2, lag) for lag in range(-timeStep_mul_FPS, timeStep_mul_FPS + 1)]

        data_size = timeStep_mul_FPS * 2
        data_ticks = []
        data_labels = []
        for i in range(0, 7):
            if i == 0:
                data_ticks.append(int(0))
                data_labels.append(int(-data_size / 2))
            else:
                data_ticks.append(int(data_ticks[i - 1] + (data_size / 6)))
                data_labels.append(int(data_labels[i - 1] + (data_size / 6)))

        offset = np.ceil(len(rs) / 2) - np.argmax(rs)
        self._SC_dictMethods[SC_TIME_LAGGED_CROSS_CORRELATION][SC_TIME_LAGGED_CROSS_CORRELATION] = offset
        _, ax = plt.subplots(figsize=(14, 5))
        ax.plot(rs)
        ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
        ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
        ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[-1.0, 1.0], xlim=[0, data_size],
               xlabel='Offset', ylabel='Pearson r')
        ax.set_xticks(data_ticks)
        ax.set_xticklabels(data_labels)
        plt.legend()

        final_export_path = os.path.normpath(exportFigDirPath) + '/' + 'FigNo02_' + exportFigFileName + \
                            '_TimeLaggedCrossCorrelation.png'
        plt.savefig(final_export_path)

        plt.close()

    def method_TimeLaggedCrossCorrelationNoSplits(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        pass

    def method_RollingWindowTimeLaggedCrossCorrelation(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        pass

    def method_DynamicTimeWarping(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        pass

    def signComp_exec_(self, arrData1: np.ndarray, arrData2: np.ndarray, exportFigDirPath: str, exportFigFileName: str):
        for _method_ in self._SC_dictMethods.keys():
            if self._SC_dictMethods[_method_][SC_EXEC_STATE]:
                self._SC_dictMethods[_method_][SC_EXEC_FUNC](arrData1, arrData2, exportFigDirPath, exportFigFileName)