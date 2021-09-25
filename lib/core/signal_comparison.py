# The analytic tutorial can be found in:
# https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
# This code was developrd by Jin Hyun Cheong

# In this code three ways to measure synchrony between time series data are presented:
# Pearson correlation, time lagged cross correlations, and dynamic time warping.
import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import dtw

plt.ioff()


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


def M_Pearson(dataArr, baseIndex, corrIndex, baseIndex_Label=None, corrIndex_Label=None,
              r_window_size=120, bool_plt_show=False, bool_plt_save=False,
              str_plt_save_dir_path='', str_plt_save_name=''):

    if baseIndex_Label is None:
        baseIndex_Label = baseIndex
    if corrIndex_Label is None:
        corrIndex_Label = baseIndex

    # Compute Pearson with Pandas
    overall_pearson_r = dataArr.corr().iloc[0, 1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # out: Pandas computed Pearson r: 0.2058774513561943

    data1 = dataArr.dropna()[baseIndex]
    data2 = dataArr.dropna()[corrIndex]

    # Compute Pearson with Scipy
    r, p = stats.pearsonr(data1, data2)
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    # out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

    # Compute rolling window synchrony
    f, ax1 = plt.subplots(figsize=(7, 3))
    color = 'tab:red'
    data1.rolling(window=r_window_size, center=True).median().plot(ax=ax1, color=color)
    ax1.set(xlabel='Time', ylabel=baseIndex_Label)
    str_title = baseIndex.replace('_', ' ') + ' - ' + corrIndex.replace('_', ' ') + \
                '\n' f"Overall Pearson r = {np.round(overall_pearson_r, 2)}" + \
                '\n' f"Squared R = {np.round(overall_pearson_r*overall_pearson_r, 2)}" + \
                '\n' f"p = {np.round(p, 7)}"
    ax1.set(title=str_title)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set(xlabel='Time', ylabel=corrIndex_Label)
    data2.rolling(window=r_window_size, center=True).median().plot(ax=ax2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    f.tight_layout()  # otherwise the right y-label is slightly clipped

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_PearsonRollingMedian.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    # Interpolate missing data.
    df_interpolated = dataArr.interpolate()
    # Compute rolling window synchrony
    rolling_r = df_interpolated[baseIndex].rolling(window=r_window_size, center=True).corr(df_interpolated[corrIndex])
    _, ax = plt.subplots(2, 1, figsize=(14, 6))
    dataArr.rolling(window=r_window_size, center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame', ylabel='Measure and Pollutant Factor')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame', ylabel='Pearson r')
    plt.suptitle("Correlation Data and Rolling Window Correlation")

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + '_PearsonRolling_R.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    print(r)
    print(r*r)
    print(p)
    print(overall_pearson_r)
    print(overall_pearson_r * overall_pearson_r)
    print(rolling_r)

    return r, r*r, p, overall_pearson_r, overall_pearson_r*overall_pearson_r, rolling_r.min(), rolling_r.max()


# def M_Spearman(dataArr, baseIndex, corrIndex, bool_plt_show=False, bool_plt_save=False,
#               str_plt_save_dir_path='', str_plt_save_name=''):
#     def histogram_intersection(a, b):
#         v = np.minimum(a, b).sum().round(decimals=1)
#         return v
#     print(dataArr.corr(method=histogram_intersection))


def M_TimeLaggedCrossCorrelation(dataArr, baseIndex, corrIndex, time_step=5, fps=30,
                                 bool_plt_show=False, bool_plt_save=False, str_plt_save_dir_path='',
                                 str_plt_save_name=''):
    d1 = dataArr[baseIndex]
    d2 = dataArr[corrIndex]
    rs = [crosscorr(d1, d2, lag) for lag in range(-int(time_step * fps), int(time_step * fps + 1))]

    data_size = time_step * fps * 2
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
    _, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', ylim=[-1.0, 1.0], xlim=[0, data_size],
           xlabel='Offset', ylabel='Pearson r')
    ax.set_xticks(data_ticks)
    ax.set_xticklabels(data_labels)
    plt.legend()

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_TimeLaggedCrossCorrelation.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    return offset


def M_TimeLaggedCrossCorrelationNoSplits(dataArr, baseIndex, corrIndex, time_step=5, fps=30, no_splits=20,
                                         bool_plt_show=False, bool_plt_save=False, str_plt_save_dir_path='',
                                         str_plt_save_name=''):
    samples_per_split = dataArr.shape[0] / no_splits
    rss = []
    for t in range(0, no_splits):
        d1 = dataArr[baseIndex].loc[t * samples_per_split:(t + 1) * samples_per_split]
        d2 = dataArr[corrIndex].loc[t * samples_per_split:(t + 1) * samples_per_split]
        rs = [crosscorr(d1, d2, lag) for lag in range(-int(time_step * fps), int(time_step * fps + 1))]
        rss.append(rs)
    rss = pd.DataFrame(rss)

    data_size = time_step * fps * 2
    data_ticks = []
    data_labels = []
    for i in range(0, 7):
        if i == 0:
            data_ticks.append(int(0))
            data_labels.append(int(-data_size / 2))
        else:
            data_ticks.append(int(data_ticks[i - 1] + (data_size / 6)))
            data_labels.append(int(data_labels[i - 1] + (data_size / 6)))

    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(rss, cmap='RdBu_r', ax=ax)
    ax.set(title=f'Windowed Time Lagged Cross Correlation', xlim=[0, data_size],
           xlabel='Offset', ylabel='Window epochs')
    ax.set_xticks(data_ticks)
    ax.set_xticklabels(data_labels)

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_TimeLaggedCrossCorrelationNoSplits.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()


def M_RollingWindowTimeLaggedCrossCorrelation(dataArr, baseIndex, corrIndex, time_step=5, fps=30,
                                              step_size=30, bool_plt_show=False, bool_plt_save=False,
                                              str_plt_save_dir_path='', str_plt_save_name=''):

    # frequency = 5400
    frequency = 500
    t_start = 0
    t_end = t_start + (time_step * fps * 2)
    rss = []
    while t_end < frequency:
        d1 = dataArr[baseIndex].iloc[t_start:t_end]
        d2 = dataArr[corrIndex].iloc[t_start:t_end]
        rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(-int(time_step * fps), int(time_step * fps + 1))]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    data_size = time_step * fps * 2
    data_ticks = []
    data_labels = []
    for i in range(0, 7):
        if i == 0:
            data_ticks.append(int(0))
            data_labels.append(int(-data_size / 2))
        else:
            data_ticks.append(int(data_ticks[i - 1] + (data_size / 6)))
            data_labels.append(int(data_labels[i - 1] + (data_size / 6)))

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rss, cmap='RdBu_r', ax=ax)
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlim=[0, data_size],
           xlabel='Offset', ylabel='Epochs')
    ax.set_xticks(data_ticks)
    ax.set_xticklabels(data_labels)

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_RollingWindowTimeLaggedCrossCorrelation.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()


def M_DynamicTimeWrapping(dataArr, baseIndex, corrIndex, bool_plt_show=False, bool_plt_save=False,
                          str_plt_save_dir_path='', str_plt_save_name=''):
    d1 = dataArr[baseIndex].interpolate().values
    d2 = dataArr[corrIndex].interpolate().values

    alignment = dtw.dtw(d1, d2, keep_internals=True)

    # Display the warping curve, i.e. the alignment curve
    alignment.plot(type="threeway")
    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_DynamicTimeWrappingThreewayAlignment.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    # Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw.dtw(d1, d2, keep_internals=True,
            step_pattern=dtw.rabinerJuangStepPattern(6, "c")).plot(type="twoway", offset=-2)

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_DynamicTimeWrappingTwowayRabinerJuangStepPattern.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    # See the recursion relation, as formula and diagram
    print(dtw.rabinerJuangStepPattern(6, "c"))
    dtw.rabinerJuangStepPattern(6, "c").plot()

    if bool_plt_save:
        final_export_path = os.path.normpath(str_plt_save_dir_path) + '/' + str_plt_save_name + \
                            '_DynamicTimeWrappingRabinerPlotCJuangStepPattern.png'
        plt.savefig(final_export_path)
    if bool_plt_show:
        plt.show()
    else:
        plt.close()

    return alignment.distance


def RunAllCorrelationMethods(dataArr, baseIndex, corrIndex, baseIndex_Label=None, corrIndex_Label=None,
                             r_window_size=120, time_step=5, fps=30, no_splits=20,
                             bool_plt_show=False, bool_plt_save=False, str_plt_save_dir_path='',
                             str_plt_save_name=''):
    # ----------------- #
    # 1) Pearson Method #
    # ----------------- #
    r, R2, p, overall_pearson_r, overall_pearson_R2, rolling_r_min, rolling_r_max = \
        M_Pearson(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex,
                  baseIndex_Label=baseIndex_Label, corrIndex_Label=corrIndex_Label,
                  r_window_size=r_window_size, bool_plt_show=bool_plt_show,
                  bool_plt_save=bool_plt_save, str_plt_save_dir_path=str_plt_save_dir_path,
                  str_plt_save_name=str_plt_save_name)

    # ----------------- #
    # 2) Spearman Method #
    # ----------------- #
    # M_Spearman(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex,
    #            bool_plt_save=bool_plt_save, str_plt_save_dir_path=str_plt_save_dir_path,
    #            str_plt_save_name=str_plt_save_name)

    # ------------------------------------------------------------ #
    # 3) Time Lagged Cross Correlation — assessing signal dynamics #
    # ------------------------------------------------------------ #
    offset = M_TimeLaggedCrossCorrelation(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex,
                                          time_step=time_step, fps=fps, bool_plt_show=bool_plt_show,
                                          bool_plt_save=bool_plt_save, str_plt_save_dir_path=str_plt_save_dir_path,
                                          str_plt_save_name=str_plt_save_name)

    # # ------------------------------------------------------------ #
    # # 4) Time Lagged Cross Correlation — assessing signal dynamics #
    # # ------------------------------------------------------------ #
    M_TimeLaggedCrossCorrelationNoSplits(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex, no_splits=no_splits,
                                         time_step=time_step, fps=fps, bool_plt_show=bool_plt_show,
                                         bool_plt_save=bool_plt_save, str_plt_save_dir_path=str_plt_save_dir_path,
                                         str_plt_save_name=str_plt_save_name)

    # # ----------------------------------------------- #
    # # 5) Rolling window time lagged cross correlation #
    # # ----------------------------------------------- #
    M_RollingWindowTimeLaggedCrossCorrelation(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex,
                                              time_step=time_step, fps=fps,
                                              bool_plt_show=bool_plt_show, bool_plt_save=bool_plt_save,
                                              str_plt_save_dir_path=str_plt_save_dir_path,
                                              str_plt_save_name=str_plt_save_name)

    # # ------------------------------------------------------------------ #
    # # 6) Dynamic Time Wrapping — synchrony of signals varying in lengths #
    # # ------------------------------------------------------------------ #
    alignment_distance = M_DynamicTimeWrapping(dataArr=dataArr, baseIndex=baseIndex, corrIndex=corrIndex,
                                               bool_plt_show=bool_plt_show, bool_plt_save=bool_plt_save,
                                               str_plt_save_dir_path=str_plt_save_dir_path,
                                               str_plt_save_name=str_plt_save_name)

    return r, R2, p, overall_pearson_r, overall_pearson_R2, rolling_r_min, rolling_r_max, offset, alignment_distance


if __name__ == '__main__':
    df = pd.read_csv('data/synchrony_sample.csv')
    RunAllCorrelationMethods(dataArr=df, baseIndex='S1_Joy', corrIndex='S2_Joy', bool_plt_show=False,
                             bool_plt_save=True, str_plt_save_dir_path='exportPlots/')
