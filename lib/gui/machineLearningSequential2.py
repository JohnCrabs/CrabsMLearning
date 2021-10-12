import sys
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import sklearn
import openpyxl as op
import time

# from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, \
    QListWidget, QListWidgetItem, QLabel
# from PySide2.QtGui import QIcon, QPixmap

# import lib.core.file_manipulation as file_manip
import lib.core.my_calendar_v2 as my_cal_v2
import lib.core.multi_output_regression as mor
import lib.core.signal_comparison as signcomp
from lib.core.project_flags import *

import lib.gui.machineLearningMainWidget as ML_mainWid
from lib.gui.guiStyle import setStyle_
import lib.gui.commonFunctions as coFunc


class WidgetMachineLearningSequential2(ML_mainWid.WidgetMachineLearningMainWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
        super().__init__(w, h, minW, minH, maxW, maxH, winTitle, iconPath)

    def actionButtonExecute(self):
        dict_list_input_columns = {}
        dict_list_output_columns = {}
        dict_primary_event_column = {}
        dict_data_storing = {}
        dict_max_values_for_input_columns = {}
        dict_max_values_for_output_columns = {}
        dict_store_folderFileNames = {}

        sequenceStepIndex = self.dict_machineLearningParameters[self.dkey_mlpTypeIndex()]
        list_of_input_headers = []
        list_of_output_headers = []
        list_of_output_headers_real = []
        list_of_output_headers_pred = []

        DKEY_INPUT = 'INPUT'
        DKEY_OUTPUT = 'OUTPUT'
        DKEY_DATA = 'DATA'
        DKEY_TRAIN = 'TRAIN'
        DKEY_TEST = 'TEST'
        DKEY_ALL = 'ALL_DATA'
        dict_dataset_categorized_by_event = {}
        dict_sequential_dataset = {}

        sequenceTestPercentage = self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()]
        export_folder_path = self.dict_machineLearningParameters[
                                 self.dkey_mlpExportFolder()] + '/' + dt.datetime.now().strftime("%d%m%Y_%H%M%S")
        file_manip.checkAndCreateFolders(export_folder_path)
        holdout_size = self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentage()]
        number_of_experiments = self.dict_machineLearningParameters[self.dkey_mlpExperimentNumber()]

        r_window_size = 15
        time_step = 2  # seconds
        fps = 30
        no_splits = 5
        dir_plot_export = 'ExportPlots'

        # ******************************************************************************************************* #

        def export_dict_with_denormalized_values(key_fileName):
            tmp_export_file_input = [['COLUMN_NAME', 'DENORMALIZED_VALUE']]
            tmp_export_file_output = [['COLUMN_NAME', 'DENORMALIZED_VALUE']]
            for in_key in dict_max_values_for_input_columns[key_fileName]:
                tmp_export_file_input.append([in_key, dict_max_values_for_input_columns[key_fileName][in_key]])

            for in_key in dict_max_values_for_output_columns[key_fileName]:
                tmp_export_file_output.append([in_key, dict_max_values_for_output_columns[key_fileName][in_key]])

            tmp_dir_folder_dataFile = export_folder_path + '/' + dict_store_folderFileNames[key_fileName] + '/Data'
            file_manip.checkAndCreateFolder(tmp_dir_folder_dataFile)
            my_cal_v2.write_csv(tmp_dir_folder_dataFile + '/Denormalize_Input_Values.csv',
                                tmp_export_file_input, my_cal_v2.del_comma)
            my_cal_v2.write_csv(tmp_dir_folder_dataFile + '/Denormalize_Output_Values.csv',
                                tmp_export_file_output, my_cal_v2.del_comma)

        def export_train_test_files(dataFileList, export_path, header_list=None):
            tmp_export_file = []
            if header_list is not None:
                tmp_export_file.append(header_list)
            for row in dataFileList:
                tmp_export_file.append(row)
            my_cal_v2.write_csv(export_path, tmp_export_file, my_cal_v2.del_comma)

        # ******************************************************************************************************* #

        if self.dict_tableFilesPaths.keys().__len__() >= 1:  # if there are at least 2 files (safety if)
            # Error Checking and store information
            for fileName in self.dict_tableFilesPaths.keys():  # for each file in tableFilePaths
                if self.BE_errorExist(fileName):  # if errors exists
                    return  # exit the function
            a = 9
            if a == 1:
                self.BE_linearExecution()
            else:
                self.BE_parallelExecution()

                list_of_primary_events = []
                dict_list_input_columns[fileName] = self.dict_tableFilesPaths[fileName][self._dkeyInputList()]
                dict_list_output_columns[fileName] = self.dict_tableFilesPaths[fileName][self._dkeyOutputList()]

                file_manip.checkAndCreateFolders(export_folder_path)

                dict_primary_event_column[fileName] = self.dict_tableFilesPaths[fileName][self.dkeyPrimaryEventColumn()]
                list_of_primary_events.append(dict_primary_event_column[fileName])

                dict_data_storing[fileName] = pd.read_csv(self.dict_tableFilesPaths[fileName][self.dkeyFullPath()])
                dict_data_storing[fileName].fillna(method='ffill', inplace=True)
                dict_data_storing[fileName].fillna(method='bfill', inplace=True)

                dict_max_values_for_input_columns[fileName] = {}
                dict_max_values_for_output_columns[fileName] = {}

                for inp_column in dict_list_input_columns[fileName]:
                    dict_max_values_for_input_columns[fileName][inp_column] = dict_data_storing[fileName][
                        inp_column].max()

                for out_column in dict_list_output_columns[fileName]:
                    dict_max_values_for_output_columns[fileName][out_column] = dict_data_storing[fileName][
                        out_column].max()

            # print(dict_list_input_columns)
            # print(dict_list_output_columns)
            print(dict_max_values_for_input_columns)
            print(dict_max_values_for_output_columns)

            # ******************************************************************************************************* #

            # Set the output headers
            for index in range(0, sequenceStepIndex):
                for fileName in self.dict_tableFilesPaths.keys():
                    for inp_column in dict_list_input_columns[fileName]:
                        list_of_input_headers.append(inp_column + "_SEQ_" + str(index))
                    for out_column in dict_list_output_columns[fileName]:
                        list_of_output_headers.append(out_column + "_SEQ_" + str(index))
                        list_of_output_headers_real.append(out_column + "_SEQ_" + str(index) + "_REAL")
                        list_of_output_headers_pred.append(out_column + "_SEQ_" + str(index) + "_PRED")

            # Check if the user has set a Primary Event or not and follow the specified path
            if list_of_primary_events.count(None) != len(list_of_primary_events):
                if list_of_primary_events.count(None) != 0:
                    print("ERROR: All files must have selected primary event")
                    return
                unique_list_of_common_primary_events = []
                for fileName in self.dict_tableFilesPaths.keys():
                    for event in dict_data_storing[fileName][dict_primary_event_column[fileName]]:
                        if event not in unique_list_of_common_primary_events:
                            unique_list_of_common_primary_events.append(event)
                # print(unique_list_of_common_primary_events)
                print("Found ", unique_list_of_common_primary_events.__len__(), " unique events!")

                # Create folders for storing the output data
                for fileName in self.dict_tableFilesPaths.keys():
                    tmp_fName = fileName.split('.')[0]
                    file_manip.checkAndCreateFolders(export_folder_path + '/' + tmp_fName)
                    dict_store_folderFileNames[fileName] = tmp_fName

                # Categorized the data by event
                print("Categorize data by selected Primary Event for each dataset...")
                for fileName in self.dict_tableFilesPaths.keys():
                    dict_dataset_categorized_by_event[fileName] = {}
                    dict_dataset_categorized_by_event[fileName][DKEY_DATA] = {}
                    for uniq_event in unique_list_of_common_primary_events:
                        dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event] = {}
                        tmp_final_arr_input = []
                        tmp_final_arr_output = []
                        for key_index in dict_data_storing[fileName][dict_primary_event_column[fileName]].keys():
                            if dict_data_storing[fileName][dict_primary_event_column[fileName]][key_index] \
                                    == uniq_event:
                                tmp_arr_input = []
                                tmp_arr_output = []
                                for input_column in dict_list_input_columns[fileName]:
                                    value = dict_data_storing[fileName][input_column][key_index] / \
                                            dict_max_values_for_input_columns[fileName][input_column]
                                    tmp_arr_input.append(value)

                                for output_column in dict_list_output_columns[fileName]:
                                    value = dict_data_storing[fileName][output_column][key_index] / \
                                            dict_max_values_for_output_columns[fileName][output_column]
                                    tmp_arr_output.append(value)
                                tmp_final_arr_input.append(tmp_arr_input)
                                tmp_final_arr_output.append(tmp_arr_output)

                        dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event][
                            DKEY_INPUT] = tmp_final_arr_input
                        dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event][
                            DKEY_OUTPUT] = tmp_final_arr_output

                    export_dict_with_denormalized_values(fileName)
                # print(dict_dataset_categorized_by_event)

                # Create the sequential data
                print("Create sequential dataset for each Primary Event for each dataset" +
                      "... (sequence step = ", sequenceStepIndex, " )")
                for fileName in self.dict_tableFilesPaths.keys():
                    dict_sequential_dataset[fileName] = {}
                    dict_sequential_dataset[fileName][DKEY_DATA] = {}
                    for uniq_event in unique_list_of_common_primary_events:
                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event] = {}
                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT] = []
                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT] = []
                        input_size = dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event][
                            DKEY_INPUT].__len__()
                        for uniq_index in range(0, input_size - (2 * sequenceStepIndex + 1)):
                            tmp_arr_input = []
                            tmp_arr_output = []

                            for i in range(uniq_index, uniq_index + sequenceStepIndex):
                                tmp_arr_input.extend(
                                    dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event][DKEY_INPUT][i])
                                tmp_arr_output.extend(
                                    dict_dataset_categorized_by_event[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT][
                                        i + sequenceStepIndex])
                            dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT].append(tmp_arr_input)
                            dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT].append(tmp_arr_output)
                # print(dict_sequential_dataset)

                print("Create TRAIN/TEST for each Primary Event (of each dataset) and merge the them to one " +
                      "array (for each dataset)...")
                for fileName in self.dict_tableFilesPaths.keys():
                    dict_sequential_dataset[fileName][DKEY_INPUT] = {}
                    dict_sequential_dataset[fileName][DKEY_OUTPUT] = {}
                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN] = []
                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST] = []
                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL] = []
                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN] = []
                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST] = []
                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL] = []

                    for uniq_event in unique_list_of_common_primary_events:
                        size_of_list = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT].__len__()
                        trte_slice_ = int(size_of_list * sequenceTestPercentage)

                        train_data_input = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT][
                                           :-trte_slice_]
                        test_data_input = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT][
                                          -trte_slice_:]

                        train_data_output = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT][
                                            :-trte_slice_]
                        test_data_output = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT][
                                           -trte_slice_:]

                        all_input = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_INPUT]
                        all_output = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_OUTPUT]

                        for index in range(0, train_data_input.__len__()):
                            dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN].append(train_data_input[index])
                            dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN].append(train_data_output[index])

                        for index in range(0, test_data_input.__len__()):
                            dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST].append(test_data_input[index])
                            dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST].append(test_data_output[index])

                        for index in range(0, all_input.__len__()):
                            dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL].append(all_input[index])
                            dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL].append(all_output[index])

                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA] = {}
                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][DKEY_INPUT] = np.array(
                            all_input)
                        dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][DKEY_OUTPUT] = np.array(
                            all_output)

                    dir_folder_dataFile = export_folder_path + '/' + dict_store_folderFileNames[fileName] + '/Data'
                    file_manip.checkAndCreateFolders(dir_folder_dataFile)

                    export_train_test_files(dataFileList=dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN],
                                            header_list=list_of_input_headers,
                                            export_path=dir_folder_dataFile + '/InputTrain.csv')

                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN] = np.array(
                        dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN])

                    export_train_test_files(dataFileList=dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST],
                                            header_list=list_of_input_headers,
                                            export_path=dir_folder_dataFile + '/InputTest.csv')

                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST] = np.array(
                        dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST])

                    export_train_test_files(dataFileList=dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN],
                                            header_list=list_of_output_headers,
                                            export_path=dir_folder_dataFile + '/OutputTrain.csv')

                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN] = np.array(
                        dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN])

                    export_train_test_files(dataFileList=dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST],
                                            header_list=list_of_output_headers,
                                            export_path=dir_folder_dataFile + '/OutputTest.csv')

                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST] = np.array(
                        dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST])

                    dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL] = np.array(
                        dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL])
                    dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL] = np.array(
                        dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL])

                    print("TRAIN__INPUT_SHAPE = ", dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN].shape)
                    print("TEST__OUTPUT_SHAPE = ", dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST].shape)
                    print("TRAIN_OUTPUT_SHAPE = ", dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN].shape)
                    print("TEST__OUTPUT_SHAPE = ", dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST].shape)
                # print(dict_sequential_dataset)

                # Perform the Experiments (MACHINE LEARNING TRAIN/TESTS)

                tmp_header_for_cor = ['Method']
                for column_name in list_of_output_headers:
                    tmp_header_for_cor.append(column_name)

                print("MACHINE LEARNING EXEC")
                for fileName in self.dict_tableFilesPaths.keys():
                    for exp_index_ in range(0, number_of_experiments):
                        print("RUN FOR ", fileName, " -> EXPERIMENT_" + str(exp_index_))
                        train_size = dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN].shape[0]
                        trainIdx = np.random.permutation(train_size)
                        trainIdx = trainIdx[:int((1.0 - holdout_size) * train_size)]

                        df_x_train_val = dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TRAIN]
                        df_y_train_val = dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TRAIN]

                        df_x_test = dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_TEST]
                        df_y_test = dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_TEST]

                        dir_folder_dataFile = export_folder_path + '/' + dict_store_folderFileNames[fileName]

                        list_models, dir_path, workbook_dirpath = mor.MachineLearning_Sequential(
                            df_x_train_val[trainIdx],
                            df_y_train_val[trainIdx],
                            df_x_test, df_y_test,
                            name='MachineLearning',
                            path=dir_folder_dataFile,
                            dnn_LactFunc='sigmoid',
                            dnn_OactFunc='tanh', dnn_loss='mse',
                            lstm_LactFunc='relu',
                            lstm_DactFunc='tanh',
                            css_LactFunc='elu',
                            lstm_loss='mean_squared_error',
                            lstm_optimizer='adam',
                            seq_div=sequenceStepIndex,
                            epochs=100,
                            batch_size=100,
                            min_lr=0.001, dropout_percentage=0.1)

                        workbook_path = workbook_dirpath + '/' + dict_store_folderFileNames[fileName] + '_Errors.xlsx'
                        for uniq_event in unique_list_of_common_primary_events:
                            size_of_list = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][
                                DKEY_INPUT].__len__()
                            trte_slice_ = int(size_of_list * sequenceTestPercentage)
                            vxLine = size_of_list - trte_slice_
                            try:
                                wb = op.load_workbook(workbook_path)
                                ws = wb.worksheets[0]  # select first worksheet
                            except FileNotFoundError:
                                headers_row = ['Event', 'Technique']
                                for count_index in range(0, sequenceStepIndex):
                                    for column_name in dict_list_output_columns[fileName]:
                                        headers_row.append(column_name + '_MAX_NORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MAX_DENORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MIN_NORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MIN_DENORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MSE_NORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MSE_DENORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_RMSE_NORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_RMSE_DENORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MAE_NORM_SEQ_' + str(count_index))
                                        headers_row.append(column_name + '_MAE_DENORM_SEQ_' + str(count_index))

                                wb = op.Workbook()
                                ws = wb.active
                                ws.append(headers_row)
                                wb.save(workbook_path)

                            df_x = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][DKEY_INPUT]
                            df_y = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][DKEY_OUTPUT]

                            inputData = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][
                                DKEY_INPUT].copy()
                            outputData = dict_sequential_dataset[fileName][DKEY_DATA][uniq_event][DKEY_DATA][
                                DKEY_OUTPUT].copy()

                            inputSeqData = inputData.reshape(inputData.shape[0], sequenceStepIndex,
                                                             int(inputData.shape[1] / sequenceStepIndex))
                            outputSeqData = outputData.reshape(outputData.shape[0], sequenceStepIndex,
                                                               int(outputData.shape[1] / sequenceStepIndex))

                            dict_models = {}
                            for m_path in list_models:
                                print(m_path)
                                if m_path.__contains__(mor.FLAG_S2S_LSTM):
                                    model = mor.loadModel(m_path)
                                    df_y_pred = mor.predModel(model, inputSeqData)
                                    o_shape = outputSeqData.shape
                                    dict_models[m_path] = [outputSeqData.reshape(o_shape[0], o_shape[1] * o_shape[2]),
                                                           df_y_pred.reshape(o_shape[0], o_shape[1] * o_shape[2])]
                                elif m_path.__contains__(mor.FLAG_LSTM):
                                    model = mor.loadModel(m_path)
                                    df_y_pred = mor.predModel(model, np.expand_dims(df_x, axis=2))
                                    dict_models[m_path] = [df_y, np.squeeze(df_y_pred, axis=2)]
                                else:
                                    model = mor.loadModel(m_path)
                                    df_y_pred = mor.predModel(model, df_x)
                                    dict_models[m_path] = [df_y, df_y_pred]

                            cor_CSV = [tmp_header_for_cor]
                            for key in dict_models.keys():
                                for model_name in mor.LIST_WITH_MODEL_FLAGS:
                                    if key.__contains__(model_name):
                                        tmp_append_row = [uniq_event, model_name]
                                        d1 = pd.DataFrame(dict_models[key][0], columns=list_of_output_headers_real)
                                        d2 = pd.DataFrame(dict_models[key][1], columns=list_of_output_headers_pred)

                                        o_dir = os.path.normpath(dir_path) + '/../' + dir_plot_export + '/' + \
                                                model_name + '/' + uniq_event + '/'
                                        file_manip.checkAndCreateFolders(o_dir)

                                        o_dir_Corr = o_dir + 'SignalCompare'
                                        file_manip.checkAndCreateFolders(o_dir_Corr)

                                        o_dir_MLP = o_dir + 'RealPredictPlots'
                                        file_manip.checkAndCreateFolders(o_dir_MLP)

                                        tmp_cor_csv = [model_name]

                                        for index in range(0, list_of_output_headers_real.__len__()):
                                            mul_ind = 1.0
                                            dataset_real = list_of_output_headers_real[index]
                                            dataset_pred = list_of_output_headers_pred[index]
                                            dataset = dataset_real
                                            dataset.replace('_REAL', '')

                                            for out_column in dict_list_output_columns[fileName]:
                                                if dataset_real.__contains__(out_column):
                                                    mul_ind = dict_max_values_for_output_columns[fileName][out_column]
                                            tmp_d1 = d1[dataset_real] * mul_ind
                                            tmp_d2 = d2[dataset_pred] * mul_ind

                                            d_cor = pd.DataFrame(np.array([tmp_d1.values, tmp_d2.values]).T,
                                                                 columns=[dataset_real, dataset_pred])
                                            (d_cor_r, d_cor_R2, d_cor_p,
                                             d_cor_overall_pearson_r, d_cor_overall_pearson_R2, d_cor_rolling_r_min,
                                             d_cor_rolling_r_max, d_cor_offset,
                                             d_cor_alignment_distance) = \
                                                signcomp.RunAllCorrelationMethods(dataArr=d_cor,
                                                                                  baseIndex=dataset_real,
                                                                                  corrIndex=dataset_pred,
                                                                                  baseIndex_Label='Real - Pearson r',
                                                                                  corrIndex_Label='Predicted - Pearson r',
                                                                                  r_window_size=r_window_size,
                                                                                  time_step=time_step,
                                                                                  fps=fps,
                                                                                  no_splits=no_splits,
                                                                                  bool_plt_show=False,
                                                                                  bool_plt_save=True,
                                                                                  str_plt_save_dir_path=o_dir_Corr,
                                                                                  str_plt_save_name=model_name + '_' + dataset + '_PRED_Corr')

                                            y_max_norm = max(d1[dataset_real].max(), d2[dataset_pred].max())
                                            y_max_norm += 0.1 * y_max_norm
                                            y_max_denorm = max(tmp_d1.max(), tmp_d1.max())
                                            y_max_denorm += 0.1 * y_max_denorm

                                            tmp_cor_csv.append(d_cor_R2)
                                            tmp_d1.plot(style=REAL_STYLE)
                                            tmp_d2.plot(style=PRED_STYLE)
                                            plt.gcf().set_size_inches(PLOT_SIZE_WIDTH, PLOT_SIZE_HEIGHT)
                                            plt.gcf().subplots_adjust(bottom=0.25)
                                            # plt.xticks(rotation=45, fontsize=_PLOT_FONTSIZE_TICKS)
                                            plt.yticks(fontsize=PLOT_FONTSIZE_TICKS)
                                            plt.legend(fontsize=PLOT_FONTSIZE_LEGEND, loc='best')
                                            plt.ylim(0, y_max_denorm)
                                            plt.vlines(vxLine, 0, y_max_denorm, colors='r', linestyles='dashed',
                                                       label=TRAIN_TEST_SEPARATOR)
                                            plt.title(model_name + ' ' + dataset, fontsize=PLOT_FONTSIZE_TITLE)
                                            # plt.xlabel('Date Range', fontsize=_PLOT_FONTSIZE_LABEL)
                                            plt.ylabel('Humans' + str(), fontsize=PLOT_FONTSIZE_LABEL)
                                            plt.savefig(o_dir_MLP + '/' + dataset + '.png',
                                                        dpi=PLOT_SIZE_DPI, bbox_inches='tight')
                                            # time.sleep(0.5)
                                            # plt.clf()
                                            plt.close()

                                            d1[dataset_real].plot(style=REAL_STYLE)
                                            d2[dataset_pred].plot(style=PRED_STYLE)
                                            plt.gcf().set_size_inches(PLOT_SIZE_WIDTH, PLOT_SIZE_HEIGHT)
                                            plt.gcf().subplots_adjust(bottom=0.25)
                                            # plt.xticks(rotation=45, fontsize=_PLOT_FONTSIZE_TICKS)
                                            plt.yticks(fontsize=PLOT_FONTSIZE_TICKS)
                                            plt.legend(fontsize=PLOT_FONTSIZE_LEGEND, loc='best')
                                            plt.ylim(0, y_max_norm)
                                            plt.vlines(vxLine, 0, y_max_norm, colors='r', linestyles='dashed',
                                                       label=TRAIN_TEST_SEPARATOR)
                                            plt.title(model_name + ' ' + dataset, fontsize=PLOT_FONTSIZE_TITLE)
                                            # plt.xlabel('Date Range', fontsize=_PLOT_FONTSIZE_LABEL)
                                            plt.ylabel('Humans (normalized)' + str(),
                                                       fontsize=PLOT_FONTSIZE_LABEL)
                                            plt.savefig(o_dir_MLP + '/' + dataset + '_normalized.png',
                                                        dpi=PLOT_SIZE_DPI, bbox_inches='tight')
                                            # time.sleep(0.5)
                                            # plt.clf()
                                            plt.close()

                                            norm_real = d1[dataset_real].values
                                            norm_pred = d2[dataset_pred].values

                                            denorm_real = tmp_d1.values
                                            denorm_pred = tmp_d2.values

                                            norm_err_max = np.abs(norm_real - norm_pred).max()
                                            denorm_err_max = np.abs(denorm_real - denorm_pred).max()

                                            norm_err_min = np.abs(norm_real - norm_pred).min()
                                            denorm_err_min = np.abs(denorm_real - denorm_pred).min()

                                            norm_err_mse = sklearn.metrics.mean_squared_error(norm_real, norm_pred)
                                            denorm_err_mse = sklearn.metrics.mean_squared_error(denorm_real,
                                                                                                denorm_pred)

                                            norm_err_rmse = np.sqrt(norm_err_mse)
                                            denorm_err_rmse = np.sqrt(denorm_err_mse)

                                            norm_err_mae = sklearn.metrics.mean_absolute_error(norm_real, norm_pred)
                                            denorm_err_mae = sklearn.metrics.mean_absolute_error(denorm_real,
                                                                                                 denorm_pred)

                                            tmp_append_row.append(norm_err_max)
                                            tmp_append_row.append(denorm_err_max)
                                            tmp_append_row.append(norm_err_min)
                                            tmp_append_row.append(denorm_err_min)
                                            tmp_append_row.append(norm_err_mse)
                                            tmp_append_row.append(denorm_err_mse)
                                            tmp_append_row.append(norm_err_rmse)
                                            tmp_append_row.append(denorm_err_rmse)
                                            tmp_append_row.append(norm_err_mae)
                                            tmp_append_row.append(denorm_err_mae)

                                        cor_CSV.append(tmp_cor_csv)
                                        ws.append(tmp_append_row)
                                        wb.save(workbook_path)
                                        time.sleep(1)

                            o_file = os.path.normpath(dir_path + "/../") + '/Correlation_R2.csv'
                            my_cal_v2.write_csv(o_file, cor_CSV)

            else:
                pass







class WidgetTabInputOutput(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())  # set the tab style

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonInputColumn = QPushButton("Add Input Column (X)")
        self.buttonInputColumn.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonInputColumn.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonInputColumn.setShortcut("I")  # Set Shortcut
        self.buttonInputColumn.setToolTip('Set selected column as Input Column.')  # Add Description

        self.buttonRemInputColumn = QPushButton("Remove")
        self.buttonRemInputColumn.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemInputColumn.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemInputColumn.setToolTip('Remove selected column from Input List.')  # Add Description

        self.buttonOutputColumn = QPushButton("Add Output Column (Y)")
        self.buttonOutputColumn.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonOutputColumn.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonOutputColumn.setShortcut("O")  # Set Shortcut
        self.buttonOutputColumn.setToolTip('Set selected column as Output Column.')  # Add Description

        self.buttonRemOutputColumn = QPushButton("Remove")
        self.buttonRemOutputColumn.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemOutputColumn.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemOutputColumn.setToolTip('Remove selected column from Output List.')  # Add Description

        self.buttonPrimaryEvent = QPushButton("Primary Event")
        self.buttonPrimaryEvent.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonPrimaryEvent.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonPrimaryEvent.setShortcut("P")  # Set Shortcut
        self.buttonPrimaryEvent.setToolTip('Set selected column as Primary Event.')  # Add Description

        self.buttonRemPrimaryEvent = QPushButton("Remove")
        self.buttonRemPrimaryEvent.setMinimumWidth(INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemPrimaryEvent.setMinimumHeight(INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemPrimaryEvent.setToolTip('Remove selected column from Primary Event.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_InputColumns = QListWidget()
        self.listWidget_InputColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_OutputColumns = QListWidget()
        self.listWidget_OutputColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_PrimaryEvent = QListWidget()
        self.listWidget_PrimaryEvent.setSelectionMode(QListWidget.ExtendedSelection)

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        # Set column/remove buttons in vbox
        hbox_listInputButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listInputButtons.addWidget(self.buttonInputColumn)  # Add buttonDate
        hbox_listInputButtons.addWidget(self.buttonRemInputColumn)  # Add buttonRemove

        hbox_listOutputButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listOutputButtons.addWidget(self.buttonOutputColumn)  # Add buttonTime
        hbox_listOutputButtons.addWidget(self.buttonRemOutputColumn)  # Add buttonRemove

        hbox_listPrimaryButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listPrimaryButtons.addWidget(self.buttonPrimaryEvent)  # Add buttonTime
        hbox_listPrimaryButtons.addWidget(self.buttonRemPrimaryEvent)  # Add buttonRemove

        # Set Input hbox
        labelInputList = QLabel("Input Columns:\n" +
                                "(the columns that will be used as training/test/validation " +
                                "input\nfor the machine learning)")
        vbox_listInputColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listInputColumns.addWidget(labelInputList)  # Add Label
        vbox_listInputColumns.addWidget(self.listWidget_InputColumns)  # Add Column List
        vbox_listInputColumns.addLayout(hbox_listInputButtons)  # Add Layout

        # Set Output hbox
        labelOutputList = QLabel("Output Columns:\n" +
                                 "(the columns that will be used as training/test/validation " +
                                 "output\nfor the machine learning)")
        vbox_listOutputColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listOutputColumns.addWidget(labelOutputList)  # Add Label
        vbox_listOutputColumns.addWidget(self.listWidget_OutputColumns)  # Add Column List
        vbox_listOutputColumns.addLayout(hbox_listOutputButtons)  # Add Layout

        # Set Primary Event hbox
        labelPrimaryEventList = QLabel("Primary/Common Event Column (Optional):")
        vbox_listPrimaryEvent = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listPrimaryEvent.addWidget(labelPrimaryEventList)  # Add Label
        vbox_listPrimaryEvent.addWidget(self.listWidget_PrimaryEvent)  # Add Column List
        vbox_listPrimaryEvent.addLayout(hbox_listPrimaryButtons)  # Add Layout

        # Set ListWidget in hbox
        hbox_listWidget = QHBoxLayout()  # Create Horizontal Layout
        hbox_listWidget.addLayout(vbox_listInputColumns)  # Add vbox_Combine_1 layout
        hbox_listWidget.addLayout(vbox_listOutputColumns)  # Add vbox_Combine_2 layout

        # Set a vbox
        vbox_finalListWidget = QVBoxLayout()
        vbox_finalListWidget.addLayout(hbox_listWidget)
        vbox_finalListWidget.addLayout(vbox_listPrimaryEvent)

        self.vbox_main_layout.addLayout(vbox_finalListWidget)


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetMachineLearningSequential2(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                                 winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')