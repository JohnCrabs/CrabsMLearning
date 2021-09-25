import sys
import os
import pandas as pd
import numpy as np
import datetime as dt
import tkinter as tk
import matplotlib.pyplot as plt

from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget, QListWidgetItem, QFileDialog, QLabel, QTabWidget
from PySide2.QtGui import QIcon, QPixmap

import lib.core.file_manipulation as file_manip
import lib.core.my_calendar_v2 as my_cal_v2
import lib.core.multi_output_regression as mor
import lib.core.signal_comparison as signcomp

_NEW_PROJECT_DEFAULT_FOLDER = file_manip.PATH_HOME
_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width
_INT_ADD_REMOVE_BUTTON_SIZE = 48

_ICON_ADD = _PROJECT_FOLDER + "/icon/add_cross_128x128.png"
_ICON_REMOVE = _PROJECT_FOLDER + "/icon/remove_line_128x128.png"
# _ICON_ADD = _PROJECT_FOLDER + "/icon/add_cross_128x128_filled.png"
# _ICON_REMOVE = _PROJECT_FOLDER + "/icon/remove_line_128x128_filled.png"

_DKEY_FILE_NAME = 'name'
_DKEY_FULLPATH = 'full-path'
_DKEY_COLUMNS = 'columns'
_DKEY_INPUT_LIST = 'input-list'
_DKEY_OUTPUT_LIST = 'output-list'
_DKEY_PRIMARY_EVENT_COLUMN = 'primary-event'

_DKEY_MLP_SEQUENCE_STEP_INDEX = 'sequence-index'
_DKEY_MLP_TEST_PERCENTAGE = 'test-percentage'
_DKEY_MLP_VALIDATION_PERCENTAGE = 'validation-percentage'
_DKEY_MLP_EXPORT_FOLDER = 'export-folder'
_DKEY_MLP_HOLDOUT_SIZE = 'holdout-size'
_DKEY_MLP_EXPER_NUMBER = 'experiment-number'

# --- PLOT INFO --- #
_PLOT_FONTSIZE_TITLE = 25
_PLOT_FONTSIZE_TICKS = 20
_PLOT_FONTSIZE_LABEL = 22.5
_PLOT_FONTSIZE_LEGEND = 18
_PLOT_SIZE_WIDTH = 12.40
_PLOT_SIZE_HEIGHT = 12.40
_PLOT_SIZE_DPI = 100

def setStyle_():
    """
    A function to store the style format of specific Qt Structure/Class component, such us
    QListWidget, QPushButton, etc.
    :return: The style
    """
    style = """
            QListWidget {
                background-color: white;
            }

            QListWidget::item {
                color: black;
            }

            QListWidget::item:hover {
                color: grey;
                background-color: lightyellow;
            }

            QListWidget::item:selected {
                color: red;
                background-color: lightblue;
            }

            QPushButton {
                color: black;
                background-color: lightblue;
            }

            QPushButton:hover {
                background-color: lightgrey;
            }

            QPushButton:pressed {
                background-color: lightyellow;
            }
            
            QPushButton:disabled {
                background-color: grey;
            }
            """
    return style


class WidgetMachineLearningSequential(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()

        self.setStyleSheet(setStyle_())  # Set the styleSheet

        # -------------------------------- #
        # ----- Set QTabWidget ----------- #
        # -------------------------------- #

        self.mainTabWidget = QTabWidget()  # Create a Tab Widget
        self.widgetTabInputOutput = WidgetTabInputOutput()  # A tab for input output columns

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
        self.setGeometry(_INT_SCREEN_WIDTH / 4, _INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height
        self.setMaximumWidth(maxW)  # Set Window Maximum Width
        self.setMaximumHeight(maxH)  # Set Window Maximum Width

        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonAdd = QPushButton()  # Create button for Add
        self.buttonAdd.setMinimumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonAdd.setMaximumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonAdd.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonAdd.setMaximumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Height
        self.buttonAdd.setIcon(QIcon(QPixmap(_ICON_ADD)))  # Add Icon
        self.buttonAdd.setToolTip('Add table files.')  # Add Description

        self.buttonRemove = QPushButton()  # Create button for Remove
        self.buttonRemove.setMinimumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonRemove.setMaximumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonRemove.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonRemove.setMaximumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Height
        self.buttonRemove.setIcon(QIcon(QPixmap(_ICON_REMOVE)))  # Add Icon
        self.buttonRemove.setToolTip('Remove table files.')  # Add Description

        self.buttonExecute = QPushButton("Execute")
        self.buttonExecute.setMinimumWidth(0)  # Set Minimum Width
        self.buttonExecute.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonExecute.setToolTip('Run the machine learning process.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_FileList = QListWidget()  # Create a ListWidget
        self.listWidget_FileList.setMinimumWidth(300)
        self.listWidget_ColumnList = QListWidget()  # Create a ListWidget
        self.listWidget_ColumnList.setSelectionMode(QListWidget.ExtendedSelection)  # Set Extended Selection
        self.fileName = None

        # ----------------------- #
        # ----- Set Actions ----- #
        # ----------------------- #
        self.setEvents_()  # Set the events/actions of buttons, listWidgets, etc., components

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #
        self.str_pathToTheProject = _NEW_PROJECT_DEFAULT_FOLDER  # var to store the projectPath
        self.dict_tableFilesPaths = {}  # a dictionary to store the table files

        self.dict_machineLearningParameters = {_DKEY_MLP_SEQUENCE_STEP_INDEX: 7,
                                               _DKEY_MLP_TEST_PERCENTAGE: 0.25,
                                               _DKEY_MLP_VALIDATION_PERCENTAGE: 0.20,
                                               _DKEY_MLP_EXPORT_FOLDER: _PROJECT_FOLDER + '/export_folder/',
                                               _DKEY_MLP_HOLDOUT_SIZE: 0.20,
                                               _DKEY_MLP_EXPER_NUMBER: 5}

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
        A function to create the widget components into the main QWidget
        :return: Nothing
        """
        # Disable Generate Button
        self.buttonExecute.setEnabled(False)

        # Set Column vbox
        labelColumnList = QLabel("Column List:")
        vbox_listColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listColumns.addWidget(labelColumnList)  # Add Label
        vbox_listColumns.addWidget(self.listWidget_ColumnList)  # Add Column List

        # Set add/remove button in vbox
        hbox_listFileButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listFileButtons.addWidget(self.buttonAdd)  # Add buttonAdd
        hbox_listFileButtons.addWidget(self.buttonRemove)  # Add buttonRemove
        hbox_listFileButtons.addWidget(self.buttonExecute)  # Add buttonGenerate

        # Set FileList in hbox
        labelFileList = QLabel("Opened File List:")
        vbox_listFile = QVBoxLayout()  # Create a Vertical Box Layout
        vbox_listFile.addWidget(labelFileList)  # Add Label
        vbox_listFile.addWidget(self.listWidget_FileList)  # Add FileList
        vbox_listFile.addLayout(vbox_listColumns)  # Add listColumns
        vbox_listFile.addLayout(hbox_listFileButtons)  # Add vbox_listFileButtons layout

        # Set main Tab Widget
        self.widgetTabInputOutput.setWidget()  # Set the Tab File Management Widget
        self.mainTabWidget.addTab(self.widgetTabInputOutput, "Input/Output Management")  # Add it to mainTanWidget

        # Set List and Tab Widget Layout
        hbox_final_layout = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_final_layout.addLayout(vbox_listFile)  # Add the listFile layout to finalLayout
        hbox_final_layout.addWidget(self.mainTabWidget)  # Add the mainTabWidget to finalLayout

        # Set Main Layout
        self.vbox_main_layout.addLayout(hbox_final_layout)  # Add the final layout to mainLayout

    def openFileDialog(self, dialogName='Pick a File', dialogOpenAt=file_manip.PATH_HOME, dialogFilters=None,
                       dialogMultipleSelection: bool = False):
        """
        A function to open a dialog for opening files.
        :param dialogName: The dialog's name.
        :param dialogOpenAt: The path the dialog will be opened
        :param dialogFilters: The dialog's filter files
        :param dialogMultipleSelection: A boolean to tell to dialog if multiple selection is supported
        :return: True/False, dialog/None
        """
        if dialogFilters is None:  # if dialogFilter is None
            dialogFilters = ['All Files (*.*)']  # set default Value
        dialog = QFileDialog(self, dialogName)  # Open a Browse Dialog
        if dialogMultipleSelection:  # if True
            dialog.setFileMode(QFileDialog.ExistingFiles)  # Set multiple selection
        dialog.setDirectory(dialogOpenAt)  # Set default directory to the default project
        dialog.setSidebarUrls([QUrl.fromLocalFile(dialogOpenAt)])  # Open to default path
        dialog.setNameFilters(dialogFilters)  # Choose SPACE Files
        if dialog.exec_() == QFileDialog.Accepted:  # if path Accepted
            return True, dialog
        else:
            return False, None

    def updateInputList(self):
        self.widgetTabInputOutput.listWidget_InputColumns.clear()  # Clear Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_INPUT_LIST] is not []:  # if event key is not []
                for event in self.dict_tableFilesPaths[key][_DKEY_INPUT_LIST]:  # for each EVENT
                    # Add ITEM to INPUT widget
                    self.widgetTabInputOutput.listWidget_InputColumns.addItem(
                        QListWidgetItem(key + " -> " + event))

    def removeFromInputColumn(self, fileName, column):
        # Remove the specified COLUMN from INPUT_COLUMN_LIST for the specified FILE
        if column in self.dict_tableFilesPaths[fileName][_DKEY_INPUT_LIST]:
            self.dict_tableFilesPaths[fileName][_DKEY_INPUT_LIST].remove(column)

    def updateOutputList(self):
        self.widgetTabInputOutput.listWidget_OutputColumns.clear()  # Clear Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_OUTPUT_LIST] is not []:  # if event key is not []
                for event in self.dict_tableFilesPaths[key][_DKEY_OUTPUT_LIST]:  # for each EVENT
                    # Add ITEM to OUTPUT widget
                    self.widgetTabInputOutput.listWidget_OutputColumns.addItem(
                        QListWidgetItem(key + " -> " + event))

    def removeFromOutputColumn(self, fileName, column):
        # Remove the specified COLUMN from OUTPUT_COLUMN_LIST for the specified FILE
        if column in self.dict_tableFilesPaths[fileName][_DKEY_OUTPUT_LIST]:
            self.dict_tableFilesPaths[fileName][_DKEY_OUTPUT_LIST].remove(column)

    def updatePrimaryEventList(self):
        self.widgetTabInputOutput.listWidget_PrimaryEvent.clear()  # Clear Primary Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            # if primary event key is not None
            if self.dict_tableFilesPaths[key][_DKEY_PRIMARY_EVENT_COLUMN] is not None:
                # Add ITEM to PRIMARY_EVENT widget
                self.widgetTabInputOutput.listWidget_PrimaryEvent.addItem(
                    QListWidgetItem(key + " -> " + self.dict_tableFilesPaths[key][_DKEY_PRIMARY_EVENT_COLUMN]))

    def resetPrimEventColumn(self, fileName):
        # Set PRIMARY_COLUMN for the specified FILE to None
        self.dict_tableFilesPaths[fileName][_DKEY_PRIMARY_EVENT_COLUMN] = None

    # ---------------------------------- #
    # ----- Reuse Action Functions ----- #
    # ---------------------------------- #

    def addItemsToList(self, fullPath, splitter=my_cal_v2.del_comma):
        fileName = fullPath.split('/')[-1:][0]  # find the name of the file
        # Create the dictionary
        self.dict_tableFilesPaths[fileName] = {_DKEY_FILE_NAME: fileName,
                                               _DKEY_FULLPATH: fullPath,
                                               _DKEY_COLUMNS: file_manip.getColumnNames(fullPath, splitter=splitter),
                                               _DKEY_INPUT_LIST: [],
                                               _DKEY_OUTPUT_LIST: [],
                                               _DKEY_PRIMARY_EVENT_COLUMN: None
                                               }
        self.listWidget_FileList.addItem(QListWidgetItem(fileName))  # Add Item to List

    # -------------------------------- #
    # ----- Print/Show Functions ----- #
    # -------------------------------- #
    def prt_dict_tableFilePaths(self):
        """
        Print the dictionary tableFilesPaths values
        :return: Nothing
        """
        for key in self.dict_tableFilesPaths.keys():
            print("file-key: ", key)
            for sec_key in self.dict_tableFilesPaths[key].keys():
                print(str(sec_key) + ': ', self.dict_tableFilesPaths[key][sec_key])
            print()

    # ------------------ #
    # ----- Events ----- #
    # ------------------ #
    def setEvents_(self):
        # Button Events
        self.buttonAdd.clicked.connect(self.actionButtonAdd)  # buttonAdd -> clicked
        self.buttonRemove.clicked.connect(self.actionButtonRemove)  # buttonRemove -> clicked
        self.buttonExecute.clicked.connect(self.actionButtonExecute)  # buttonGenerate -> clicked

        # ListWidget Events
        self.listWidget_FileList.currentRowChanged.connect(self.actionFileListRowChanged_event)
        # buttonInputColumn
        self.widgetTabInputOutput.buttonInputColumn.clicked.connect(self.actionButtonInput)
        # buttonRemInputColumn
        self.widgetTabInputOutput.buttonRemInputColumn.clicked.connect(self.actionButtonRemInput)
        # buttonOutputColumn
        self.widgetTabInputOutput.buttonOutputColumn.clicked.connect(self.actionButtonOutput)
        # buttonRemOutputColumn
        self.widgetTabInputOutput.buttonRemOutputColumn.clicked.connect(self.actionButtonRemOutput)
        # buttonPrimaryEvent
        self.widgetTabInputOutput.buttonPrimaryEvent.clicked.connect(self.actionButtonPrimaryEvent)
        # buttonRemPrimaryEvent
        self.widgetTabInputOutput.buttonRemPrimaryEvent.clicked.connect(self.actionButtonRemPrimaryEvent)

    def actionButtonAdd(self):
        # Open a dialog for CSV files
        success, dialog = self.openFileDialog(dialogName='Open Table File (Currently strictly CSV)',
                                              dialogOpenAt=self.str_pathToTheProject,
                                              dialogFilters=["CSV File Format (*.csv)"],
                                              dialogMultipleSelection=True)

        if success:  # if True
            for filePath in dialog.selectedFiles():  # for each file in all selected files
                fileName = filePath.split('/')[-1:][0]  # get the filename
                # print(fullPath)
                # print(fileName)
                # print()
                if fileName not in self.dict_tableFilesPaths.keys():  # if file haven't added before
                    self.addItemsToList(filePath)  # add file to the table list

            if self.listWidget_FileList.currentItem() is None:  # Set row 0 as current row
                self.listWidget_FileList.setCurrentRow(0)  # Set current row

            if self.dict_tableFilesPaths.keys().__len__() >= 1:
                self.buttonExecute.setEnabled(True)
            # self.prt_dict_tableFilePaths()

    def actionButtonRemove(self):
        if self.listWidget_FileList.currentItem() is not None:  # if some item is selected
            self.dict_tableFilesPaths.pop(self.fileName, None)  # Delete item from dict
            self.listWidget_FileList.takeItem(self.listWidget_FileList.currentRow())  # Delete item from widget
            self.actionFileListRowChanged_event()  # run the row changed event
            self.updatePrimaryEventList()  # update PRIMARY_EVENT widget
            self.updateEventsList()  # update EVENT widget

            if self.dict_tableFilesPaths.keys().__len__() < 1:
                self.buttonGenerate.setEnabled(False)

    def actionButtonExecute(self):
        dict_list_input_columns = {}
        dict_list_output_columns = {}
        dict_primary_event_column = {}
        dict_data_storing = {}
        dict_max_values_for_input_columns = {}
        dict_max_values_for_output_columns = {}
        dict_store_folderFileNames = {}

        sequenceStepIndex = self.dict_machineLearningParameters[_DKEY_MLP_SEQUENCE_STEP_INDEX]
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

        sequenceTestPercentage = self.dict_machineLearningParameters[_DKEY_MLP_TEST_PERCENTAGE]
        export_folder_path = self.dict_machineLearningParameters[
                                 _DKEY_MLP_EXPORT_FOLDER] + '/' + dt.datetime.now().strftime("%d%m%Y_%H%M%S")
        file_manip.checkAndCreateFolder(export_folder_path)
        holdout_size = self.dict_machineLearningParameters[_DKEY_MLP_HOLDOUT_SIZE]
        number_of_experiments = self.dict_machineLearningParameters[_DKEY_MLP_EXPER_NUMBER]

        r_window_size = 15
        time_step = 2  # seconds
        fps = 30
        no_splits = 5
        dir_plot_export = 'ExportPlots'

        def export_dict_with_denormalized_values(key_fileName):
            tmp_export_file_input = [['COLUMN_NAME', 'DENORMALIZED_VALUE']]
            tmp_export_file_output = [['COLUMN_NAME', 'DENORMALIZED_VALUE']]
            for key in dict_max_values_for_input_columns[key_fileName]:
                tmp_export_file_input.append([key, dict_max_values_for_input_columns[key_fileName][key]])

            for key in dict_max_values_for_output_columns[key_fileName]:
                tmp_export_file_output.append([key, dict_max_values_for_output_columns[key_fileName][key]])

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

        if self.dict_tableFilesPaths.keys().__len__() >= 1:  # if there are at least 2 files (safety if)
            # Error Checking and store information
            list_of_primary_events = []
            for fileName in self.dict_tableFilesPaths.keys():
                if self.dict_tableFilesPaths[fileName][_DKEY_INPUT_LIST]:
                    dict_list_input_columns[fileName] = self.dict_tableFilesPaths[fileName][_DKEY_INPUT_LIST]
                else:
                    print("ERROR: <", fileName, "> has no INPUT columns specified!")
                    return

                if self.dict_tableFilesPaths[fileName][_DKEY_OUTPUT_LIST]:
                    dict_list_output_columns[fileName] = self.dict_tableFilesPaths[fileName][_DKEY_OUTPUT_LIST]
                else:
                    print("ERROR: <", fileName, "> has no OUTPUT columns specified!")
                    return

                file_manip.checkAndCreateFolders(export_folder_path)

                dict_primary_event_column[fileName] = self.dict_tableFilesPaths[fileName][_DKEY_PRIMARY_EVENT_COLUMN]
                list_of_primary_events.append(dict_primary_event_column[fileName])

                dict_data_storing[fileName] = pd.read_csv(self.dict_tableFilesPaths[fileName][_DKEY_FULLPATH])
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

                        list_models, dir_path = mor.MachineLearning_Sequential(df_x_train_val[trainIdx],
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
                                                                               lstm_optimizer='adam', seq_div=7,
                                                                               epochs=100,
                                                                               batch_size=100,
                                                                               min_lr=0.001, dropout_percentage=0.1)

                        df_x = dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL]
                        df_y = dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL]

                        inputData = dict_sequential_dataset[fileName][DKEY_INPUT][DKEY_ALL].copy()
                        outputData = dict_sequential_dataset[fileName][DKEY_OUTPUT][DKEY_ALL].copy()

                        inputSeqData = inputData.reshape(inputData.shape[0], 7, int(inputData.shape[1] / 7))
                        outputSeqData = outputData.reshape(outputData.shape[0], 7, int(outputData.shape[1] / 7))

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
                                    d1 = pd.DataFrame(dict_models[key][0], columns=list_of_output_headers_real)
                                    d2 = pd.DataFrame(dict_models[key][1], columns=list_of_output_headers_pred)

                                    o_dir = os.path.normpath(dir_path) + '/../' + dir_plot_export + '/' + model_name + '/'
                                    file_manip.checkAndCreateFolders(o_dir)

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
                                                                              str_plt_save_dir_path=o_dir,
                                                                              str_plt_save_name=model_name + '_' + dataset + '_PRED_Corr')

                                        tmp_cor_csv.append(d_cor_R2)
                                        tmp_d1.plot()
                                        tmp_d2.plot()
                                        plt.gcf().set_size_inches(_PLOT_SIZE_WIDTH, _PLOT_SIZE_HEIGHT)
                                        plt.gcf().subplots_adjust(bottom=0.25)
                                        # plt.xticks(rotation=45, fontsize=_PLOT_FONTSIZE_TICKS)
                                        plt.yticks(fontsize=_PLOT_FONTSIZE_TICKS)
                                        plt.legend(fontsize=_PLOT_FONTSIZE_LEGEND, loc='best')
                                        plt.ylim(0, mul_ind)
                                        plt.title(model_name + ' ' + dataset, fontsize=_PLOT_FONTSIZE_TITLE)
                                        # plt.xlabel('Date Range', fontsize=_PLOT_FONTSIZE_LABEL)
                                        plt.ylabel('Humans per Million' + str(), fontsize=_PLOT_FONTSIZE_LABEL)
                                        plt.savefig(o_dir + dataset + '.png', dpi=_PLOT_SIZE_DPI)
                                        # time.sleep(0.5)
                                        # plt.clf()
                                        plt.close()

                                        d1[dataset_real].plot()
                                        d2[dataset_pred].plot()
                                        plt.gcf().set_size_inches(_PLOT_SIZE_WIDTH, _PLOT_SIZE_HEIGHT)
                                        plt.gcf().subplots_adjust(bottom=0.25)
                                        # plt.xticks(rotation=45, fontsize=_PLOT_FONTSIZE_TICKS)
                                        plt.yticks(fontsize=_PLOT_FONTSIZE_TICKS)
                                        plt.legend(fontsize=_PLOT_FONTSIZE_LEGEND, loc='best')
                                        plt.ylim(0, 1.0)
                                        plt.title(model_name + ' ' + dataset, fontsize=_PLOT_FONTSIZE_TITLE)
                                        # plt.xlabel('Date Range', fontsize=_PLOT_FONTSIZE_LABEL)
                                        plt.ylabel('Humans per Million (normalized)' + str(),
                                                   fontsize=_PLOT_FONTSIZE_LABEL)
                                        plt.savefig(o_dir + dataset + '_normalized.png', dpi=_PLOT_SIZE_DPI)
                                        # time.sleep(0.5)
                                        # plt.clf()
                                        plt.close()

                                    cor_CSV.append(tmp_cor_csv)

                        o_file = os.path.normpath(dir_path + "/../") + '/Correlation_R2.csv'
                        my_cal_v2.write_csv(o_file, cor_CSV)

            else:
                pass

    def actionFileListRowChanged_event(self):
        self.listWidget_ColumnList.clear()  # Clear Column Widget
        if self.listWidget_FileList.currentItem() is not None:  # If current item is not None
            self.fileName = self.listWidget_FileList.currentItem().text()  # get current item name
            for column in self.dict_tableFilesPaths[self.fileName][_DKEY_COLUMNS]:  # for each column
                # Add columns as ITEMS to widget
                self.listWidget_ColumnList.addItem(QListWidgetItem(column))

            if self.listWidget_ColumnList.currentItem() is None:  # If COLUMN widget is not None
                self.listWidget_ColumnList.setCurrentRow(0)  # Set first row selected
        else:
            self.fileName = None

    def actionButtonInput(self):
        # If some file is selected and some columns are selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current columns selected
            currentSelectedItems = self.listWidget_ColumnList.selectedItems()
            for currentColumnSelected in currentSelectedItems:  # for each item selected
                # if this column is not in the INPUT List
                if currentColumnSelected.text() not in self.dict_tableFilesPaths[self.fileName][_DKEY_INPUT_LIST]:
                    # Add it to list
                    self.dict_tableFilesPaths[self.fileName][_DKEY_INPUT_LIST].append(currentColumnSelected.text())
                # print(currentFileName, " -> ", currentColumnSelected.text())
            self.updateInputList()  # update Event widget

    def actionButtonRemInput(self):
        # If some file is selected and some columns are selected
        if self.widgetTabInputOutput.listWidget_InputColumns.currentItem() is not None:
            # get selected items
            selectedItems = self.widgetTabInputOutput.listWidget_InputColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                columnName = tmp_str.split(' -> ')[1]  # get columnName
                self.removeFromInputColumn(fileName, columnName)  # remove event from the list
            self.updateInputList()  # update EVENT widget

    def actionButtonOutput(self):
        # If some file is selected and some columns are selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current columns selected
            currentSelectedItems = self.listWidget_ColumnList.selectedItems()
            for currentColumnSelected in currentSelectedItems:  # for each item selected
                # if this column is not in the INPUT List
                if currentColumnSelected.text() not in self.dict_tableFilesPaths[self.fileName][_DKEY_OUTPUT_LIST]:
                    # Add it to list
                    self.dict_tableFilesPaths[self.fileName][_DKEY_OUTPUT_LIST].append(currentColumnSelected.text())
                # print(currentFileName, " -> ", currentColumnSelected.text())
            self.updateOutputList()  # update Event widget

    def actionButtonRemOutput(self):
        # If some file is selected and some columns are selected
        if self.widgetTabInputOutput.listWidget_OutputColumns.currentItem() is not None:
            # get selected items
            selectedItems = self.widgetTabInputOutput.listWidget_OutputColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                columnName = tmp_str.split(' -> ')[1]  # get columnName
                self.removeFromOutputColumn(fileName, columnName)  # remove event from the list
            self.updateOutputList()  # update EVENT widget

    def actionButtonPrimaryEvent(self):
        # If some file is selected and some column is selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current column name
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            # If this column exist in the input list
            if currentColumnSelected in self.dict_tableFilesPaths[self.fileName][_DKEY_INPUT_LIST]:
                self.removeFromInputColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateInputList()  # update Input widget
            # If this column exist in the output list
            if currentColumnSelected in self.dict_tableFilesPaths[self.fileName][_DKEY_OUTPUT_LIST]:
                self.removeFromOutputColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateOutputList()  # update Input widget

            # print(currentFileName, " -> ", currentColumnSelected)

            # Add it to the PRIMARY_EVENT
            self.dict_tableFilesPaths[self.fileName][_DKEY_PRIMARY_EVENT_COLUMN] = currentColumnSelected
            self.updatePrimaryEventList()  # update Primary Event widget

    def actionButtonRemPrimaryEvent(self):
        # If some file is selected and some columns are selected
        if self.widgetTabInputOutput.listWidget_PrimaryEvent.isActiveWindow() and \
                self.widgetTabInputOutput.listWidget_PrimaryEvent.currentItem() is not None:
            # get selected item
            selectedItems = self.widgetTabInputOutput.listWidget_PrimaryEvent.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                self.resetPrimEventColumn(fileName)  # remove PRIMARY_EVENT from the list
            self.updatePrimaryEventList()  # update PRIMARY_EVENT widget


class WidgetTabInputOutput(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonInputColumn = QPushButton("Add Input Column (X)")
        self.buttonInputColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonInputColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonInputColumn.setShortcut("I")  # Set Shortcut
        self.buttonInputColumn.setToolTip('Set selected column as Input Column.')  # Add Description

        self.buttonRemInputColumn = QPushButton("Remove")
        self.buttonRemInputColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemInputColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemInputColumn.setToolTip('Remove selected column from Input List.')  # Add Description

        self.buttonOutputColumn = QPushButton("Add Output Column (Y)")
        self.buttonOutputColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonOutputColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonOutputColumn.setShortcut("O")  # Set Shortcut
        self.buttonOutputColumn.setToolTip('Set selected column as Output Column.')  # Add Description

        self.buttonRemOutputColumn = QPushButton("Remove")
        self.buttonRemOutputColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemOutputColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemOutputColumn.setToolTip('Remove selected column from Output List.')  # Add Description

        self.buttonPrimaryEvent = QPushButton("Primary Event")
        self.buttonPrimaryEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonPrimaryEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonPrimaryEvent.setShortcut("P")  # Set Shortcut
        self.buttonPrimaryEvent.setToolTip('Set selected column as Primary Event.')  # Add Description

        self.buttonRemPrimaryEvent = QPushButton("Remove")
        self.buttonRemPrimaryEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemPrimaryEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
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
    widgetWin = WidgetMachineLearningSequential(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                                winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
