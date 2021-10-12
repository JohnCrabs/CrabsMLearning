import os.path
import sys
import matplotlib
import pandas as pd
import numpy as np

from PySide2.QtCore import (
    Qt
)

from PySide2.QtWidgets import (
    QWidget,
    QApplication,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QSpacerItem,
    QListWidget,
    QListWidgetItem,
    # QFileDialog,
    QLabel,
    QTabWidget,
    # QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QScrollArea
)

from PySide2.QtGui import (
    QIcon,
    QPixmap
)

# import lib.core.my_calendar_v2 as my_cal_v2
from lib.core.project_flags import *
import lib.core.machineLearningRegression as mlr

from lib.gui.guiStyle import setStyle_
import lib.gui.commonFunctions as coFunc


# *************************************************************************************************** #

matplotlib.use("Agg")  # Set matplotlib to use non-interface (don't plot figures)


class WidgetMachineLearningMainWidget(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=None, maxH=None,
                 winTitle='My Window', iconPath=None):
        super().__init__()
        self.setStyleSheet(setStyle_())  # Set the styleSheet
        self.iconPath = iconPath

        # Set this flag to True to show debugging messages to console
        self.debugMessageFlag = True

        # -------------------------------- #
        # ----- Private QTabWidget ------- #
        # -------------------------------- #

        # DICTIONARY FILE PARAMETERS
        self._DKEY_FILE_NAME: str = 'name'
        self._DKEY_FULLPATH: str = 'full-path'
        self._DKEY_COLUMNS: str = 'columns'
        self._DKEY_INPUT_LIST: str = 'input-list'
        self._DKEY_OUTPUT_LIST: str = 'output-list'
        self._DKEY_PRIMARY_EVENT_COLUMN: str = 'primary-event'

        # DICTIONARY MACHINE LEARNING PARAMETERS
        self._DKEY_MLP_TEST_PERCENTAGE: str = 'test-percentage'
        self._DKEY_MLP_TEST_PERCENTAGE_DISTRIBUTION: str = 'test-percentage-distribution'
        self._DKEY_MLP_HOLDOUT_PERCENTAGE: str = 'holdout-percentage'
        self._DKEY_MLP_HOLDOUT_PERCENTAGE_DISTRIBUTION: str = 'holdout-percentage-distribution'
        self._DKEY_MLP_EXPORT_FOLDER: str = 'export-folder'
        self._DKEY_MLP_EXPER_NUMBER: str = 'experiment-number'
        self._DKEY_MLP_ML_METHOD: str = 'ml-method'
        self._DKEY_MLP_METHOD_INDEX: str = 'ml-type-index'
        self._DKEY_MLP_MULTIFILE_TRAINING_PROCESSING: str = 'multifile-training-processing'

        # -------------------------------- #
        # ----- Set QTabWidget ----------- #
        # -------------------------------- #

        self.mainTabWidget = QTabWidget()  # Create a Tab Widget
        self.widgetTabInputOutput = WidgetTabInputOutput()  # create a tab for input output columns
        self.widgetTabMachineLearningSettings = WidgetTabMachineLearningSettings()

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(self.iconPath))  # Set Window Icon
        self.setGeometry(INT_SCREEN_WIDTH / 4, INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height
        if maxW is not None:
            self.setMaximumWidth(maxW)  # Set Window Maximum Width
        if maxH is not None:
            self.setMaximumHeight(maxH)  # Set Window Maximum Width

        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # -------------------------- #
        # ----- Set PushButton ----- #
        # -------------------------- #
        self.buttonAdd = QPushButton()  # Create button for Add
        self.buttonAdd.setMinimumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonAdd.setMaximumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonAdd.setMinimumHeight(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonAdd.setMaximumHeight(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Height
        self.buttonAdd.setIcon(QIcon(QPixmap(ICON_ADD)))  # Add Icon
        self.buttonAdd.setToolTip('Add table files.')  # Add Description

        self.buttonRemove = QPushButton()  # Create button for Remove
        self.buttonRemove.setMinimumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonRemove.setMaximumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonRemove.setMinimumHeight(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonRemove.setMaximumHeight(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Height
        self.buttonRemove.setIcon(QIcon(QPixmap(ICON_REMOVE)))  # Add Icon
        self.buttonRemove.setToolTip('Remove table files.')  # Add Description

        self.buttonExecute = QPushButton("Execute")
        self.buttonExecute.setMinimumWidth(0)  # Set Minimum Width
        self.buttonExecute.setMinimumHeight(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonExecute.setToolTip('Run the machine learning process.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_FileList = QListWidget()  # Create a ListWidget
        self.listWidget_FileList.setMinimumWidth(300)
        self.listWidget_FileList.setMaximumWidth(500)
        self.listWidget_ColumnList = QListWidget()  # Create a ListWidget
        self.listWidget_ColumnList.setMinimumWidth(300)
        self.listWidget_ColumnList.setMaximumWidth(500)
        self.listWidget_ColumnList.setSelectionMode(QListWidget.ExtendedSelection)  # Set Extended Selection
        self.fileName = None

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #
        self.str_pathToTheProject = NEW_PROJECT_DEFAULT_FOLDER  # var to store the projectPath
        self.dict_tableFilesPaths = {}  # a dictionary to store the table files

        self.dict_machineLearningParameters = {
            self.dkey_mlpExperimentNumber(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultExperimentNumber(),
            self.dkey_mlpTestPercentage(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultTestPercentage(),
            self.dkey_mlpTestPercentageDistribution(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultTestPercentageDistribution(),
            self.dkey_mlpHoldoutPercentage(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultHoldoutPercentage(),
            self.dkey_mlpHoldoutPercentageDistribution(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultHoldoutPercentageDistribution(),
            self.dkey_mlpExportFolder(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultExportPath(),
            self.dkey_mlpMethod(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultMethod(),
            self.dkey_mlpMethodIndex(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultMethodIndex(),
            self.dkey_multifileTrainingProcessing(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultMultifileTrainingProcessing()
        }

    # DICTIONARY FILE PARAMETERS
    def dkeyFileName(self):
        return self._DKEY_FILE_NAME

    def dkeyFullPath(self):
        return self._DKEY_FULLPATH

    def dkeyColumnsList(self):
        return self._DKEY_COLUMNS

    def dkeyInputList(self):
        return self._DKEY_INPUT_LIST

    def dkeyOutputList(self):
        return self._DKEY_OUTPUT_LIST

    def dkeyPrimaryEventColumn(self):
        return self._DKEY_PRIMARY_EVENT_COLUMN

    # DICTIONARY MACHINE LEARNING PARAMETERS
    def dkey_mlpTestPercentage(self):
        return self._DKEY_MLP_TEST_PERCENTAGE

    def dkey_mlpTestPercentageDistribution(self):
        return self._DKEY_MLP_TEST_PERCENTAGE_DISTRIBUTION

    def dkey_mlpExportFolder(self):
        return self._DKEY_MLP_EXPORT_FOLDER

    def dkey_mlpHoldoutPercentage(self):
        return self._DKEY_MLP_HOLDOUT_PERCENTAGE

    def dkey_mlpHoldoutPercentageDistribution(self):
        return self._DKEY_MLP_HOLDOUT_PERCENTAGE_DISTRIBUTION

    def dkey_mlpExperimentNumber(self):
        return self._DKEY_MLP_EXPER_NUMBER

    def dkey_mlpMethod(self):
        return self._DKEY_MLP_ML_METHOD

    def dkey_mlpMethodIndex(self):
        return self._DKEY_MLP_METHOD_INDEX

    def dkey_multifileTrainingProcessing(self):
        return self._DKEY_MLP_MULTIFILE_TRAINING_PROCESSING

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setTab(self):
        # Set main Tab Widget
        self.widgetTabInputOutput.setWidget()  # Set the Tab File Management Widget
        self.mainTabWidget.addTab(self.widgetTabInputOutput, "Input/Output Management")  # Add it to mainTanWidget

    def setWidget(self):
        """
        A function to create the widget components into the main QWidget
        :return: Nothing
        """

        self.setTab()
        self.widgetTabMachineLearningSettings.setWidget()
        self.mainTabWidget.addTab(self.widgetTabMachineLearningSettings, "Machine Learning Settings")

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

        # Set List and Tab Widget Layout
        hbox_final_layout = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_final_layout.addLayout(vbox_listFile)  # Add the listFile layout to finalLayout
        hbox_final_layout.addWidget(self.mainTabWidget)  # Add the mainTabWidget to finalLayout

        # Set Main Layout
        self.vbox_main_layout.addLayout(hbox_final_layout)  # Add the final layout to mainLayout

        # ----------------------- #
        # ----- Set Actions ----- #
        # ----------------------- #
        self.setMainEvents_()  # Set the events/actions of buttons, listWidgets, etc., components

    # ---------------------------------- #
    # ----- Reuse Action Functions ----- #
    # ---------------------------------- #

    def addItemsToList(self, fullPath):
        fileName = fullPath.split('/')[-1:][0]  # find the name of the file
        # Create the dictionary
        self.dict_tableFilesPaths[fileName] = {self.dkeyFileName(): fileName,
                                               self.dkeyFullPath(): fullPath,
                                               self.dkeyColumnsList(): file_manip.getColumnNames(fullPath),
                                               self.dkeyInputList(): [],
                                               self.dkeyOutputList(): [],
                                               self.dkeyPrimaryEventColumn(): None
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
    # ***** SET EVENTS FUNCTIONS *** #
    def setEvents_(self):
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

    def setTabSettingsGeneralEvents_(self):
        # Spin Boxes Events
        self.widgetTabMachineLearningSettings.tabGeneral.spinBox_ExperimentNumber.valueChanged.connect(
            self.actionExperimentResultChange)

        self.widgetTabMachineLearningSettings.tabGeneral.spinBox_MachineLearningMethodsIndex.valueChanged.connect(
            self.actionMachineLearningMethodIndexChange)

        # Double Spin Boxes  Events
        self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_TestPercentage.valueChanged.connect(
            self.actionTestPercentageChange)

        self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_HoldoutPercentage.valueChanged.connect(
            self.actionHoldoutPercentageChange)

        # Line Edit Events
        self.widgetTabMachineLearningSettings.tabGeneral.lineEdit_SetOutPath.textChanged.connect(
            self.actionLineEditChange)

        # Button Events
        self.widgetTabMachineLearningSettings.tabGeneral.buttonSetOutPath.clicked.connect(
            self.actionButtonSetOutPathClicked)

        # Combo Box Events
        self.widgetTabMachineLearningSettings.tabGeneral.comboBox_MachineLearningMethods.currentTextChanged.connect(
            self.actionMachineLearningMethodChange)

        self.widgetTabMachineLearningSettings.tabGeneral.comboBox_MultifileTrainingProcessing.currentTextChanged.connect(
            self.actionMultifileTrainingProcessingChange)

        self.widgetTabMachineLearningSettings.tabGeneral.comboBox_TestPercentageDistribution.currentTextChanged.connect(
            self.actionTestPercentageDistributionChange)

        self.widgetTabMachineLearningSettings.tabGeneral.comboBox_HoldoutPercentageDistribution.currentTextChanged.connect(
            self.actionHoldoutPercentageDistributionChange)

    def setTabSettingsLinearRegressionEvents_(self):
        pass

    def setTabSettingsRidgeEvents_(self):
        pass

    def setTabSettingsLassoEvents_(self):
        pass

    def setTabSettingsDecisionTreeRegressorEvents_(self):
        pass

    def setTabSettingsRandomForestRegressorEvents_(self):
        pass

    def setTabSettingsGradientBoostingRegressorEvents_(self):
        pass

    def setTabSettingsAdaBoostRegressorEvents_(self):
        pass

    def setTabSettingsKNeighborsRegressorEvents_(self):
        pass

    def setMainEvents_(self):
        # Button Events
        self.buttonAdd.clicked.connect(self.actionButtonAdd)  # buttonAdd -> clicked
        self.buttonRemove.clicked.connect(self.actionButtonRemove)  # buttonRemove -> clicked
        self.buttonExecute.clicked.connect(self.actionButtonExecute)  # buttonGenerate -> clicked
        # ListWidget Events
        self.listWidget_FileList.currentRowChanged.connect(self.actionFileListRowChanged_event)

        self.setEvents_()  # set the user specified event (inherited)
        self.setTabSettingsGeneralEvents_()  # set the tab settings GENERAL events
        self.setTabSettingsLinearRegressionEvents_()  # set the tab settings LINEAR REGRESSION events
        self.setTabSettingsRidgeEvents_()  # set the tab settings RIDGE events
        self.setTabSettingsLassoEvents_()  # set the tab settings LASSO events
        self.setTabSettingsDecisionTreeRegressorEvents_()  # set the tab settings DECISION TREE REGRESSOR events
        self.setTabSettingsRandomForestRegressorEvents_()  # set the tab settings RANDOM FOREST REGRESSOR events
        self.setTabSettingsGradientBoostingRegressorEvents_()  # set the tab settings GRADIENT BOOSTING REGRESSOR events
        self.setTabSettingsAdaBoostRegressorEvents_()  # set the tab settings ADA BOOST REGRESSOR events
        self.setTabSettingsKNeighborsRegressorEvents_()  # set the tab settings K NEIGHBORS REGRESSOR events

    # -------------------------- #
    # ----- Events Actions ----- #
    # -------------------------- #
    # ***** SET MAIN EVENTS ACTIONS *** #
    def actionButtonAdd(self):
        # Open dialog
        success, dialog = coFunc.openFileDialog(
            classRef=self,
            dialogName='Open Table File',
            dialogOpenAt=self.str_pathToTheProject,
            dialogFilters=["CSV File Format (*.csv)",
                           "Microsoft Office Excel File (*.xlsx)"],
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

    def updateButtonRemove(self):
        self.updateInputList()
        self.updateOutputList()
        self.updatePrimaryEventList()

    def actionButtonRemove(self):
        if self.listWidget_FileList.currentItem() is not None:  # if some item is selected
            self.dict_tableFilesPaths.pop(self.fileName, None)  # Delete item from dict
            self.listWidget_FileList.takeItem(self.listWidget_FileList.currentRow())  # Delete item from widget
            self.actionFileListRowChanged_event()  # run the row changed event
            self.updateButtonRemove()

            # if there are not enough files loaded
            if self.dict_tableFilesPaths.keys().__len__() < 1:
                self.buttonExecute.setEnabled(False)  # disable the Execute Button

    # *********************************************************** #
    # *********** Helping Functions for ButtonExecute *********** #
    # *********************************************************** #
    # *                                                         * #

    def BE_errorExist(self, fName):
        errorType404 = "Error 404: "  # set error type
        # Check if Input Columns don't exist for the specified file
        if not self.dict_tableFilesPaths[fName][self.dkeyInputList()]:  # if True
            msg = "In < " + fName + " > no INPUT columns found!"  # set message
            coFunc.consoleMessage(errorType404 + msg)  # print message to Console as Warning
            coFunc.errorMessageDialog(classRef=self,
                                      errorType=errorType404,
                                      textMessageInfo=msg,
                                      iconPath=self.iconPath)  # print message to Error Dialog
            return True  # return True
        # Check if Output Columns don't exist for the specified file
        elif not self.dict_tableFilesPaths[fName][self.dkeyOutputList()]:  # if True
            msg = "In < " + fName + " > no OUTPUT columns found!"  # set message
            coFunc.consoleMessage(errorType404 + msg)  # print message to Console as Warning
            coFunc.errorMessageDialog(classRef=self,
                                      errorType=errorType404,
                                      textMessageInfo=msg,
                                      iconPath=self.iconPath)  # print message to Error Dialog
            return True  # return True
        return False  # return False

    def BE_readFileData(self, fName):
        # find the file suffix (extension) and take is as lowercase without the comma
        suffix = os.path.splitext(self.dict_tableFilesPaths[fName][self.dkeyFullPath()])[1].lower().split('.')[1]
        # create a variable named fileData and set it to None (it will store panda array if all goes well)
        fileData = None
        # create a list to store all the needed columns (we don't need columns we will not use - RAM saving)
        fileDataUseColumns = []
        # Check if user selected primary event column
        if self.dict_tableFilesPaths[fName][self.dkeyPrimaryEventColumn()] is not None:
            fileDataUseColumns.append(self.dict_tableFilesPaths[fName][self.dkeyPrimaryEventColumn()])  # add it to list
            # print(self.dict_tableFilesPaths[fName][self.dkeyPrimaryEventColumn()])
        # append to list (if not already) the values from input column list
        [fileDataUseColumns.append(col) for col in self.dict_tableFilesPaths[fName][self.dkeyInputList()]
         if col not in fileDataUseColumns]
        # append to list (if not already) the values from output column list
        [fileDataUseColumns.append(col) for col in self.dict_tableFilesPaths[fName][self.dkeyOutputList()]
         if col not in fileDataUseColumns]
        # print(fileDataUseColumns)

        print("Read " + suffix + " file.")  # info message

        # Read the file Data - CSV
        if suffix == 'csv':
            # read only the needed columns
            fileData = pd.read_csv(self.dict_tableFilesPaths[fName][self.dkeyFullPath()])[fileDataUseColumns]
            # print(fileData.keys())
        elif suffix == 'xlsx':
            # read only the needed columns
            fileData = pd.read_excel(self.dict_tableFilesPaths[fName][self.dkeyFullPath()])[fileDataUseColumns]

        # print(fileData)

        if fileData is not None:  # if file has been read successfully
            fileData.fillna(method='ffill', inplace=True)  # fill na values using ffill
            fileData.fillna(method='bfill', inplace=True)  # fill na values using bfill
        return fileData  # return the fileData

    def BE_setTrainValTestArrays(self, dictDataInput: {}, dictDataOutput: {}, inputHeaders: [], outputHeaders: []):
        def getTrainValTest(inputArr, outputArr):
            datasetSize = inputArr.__len__()
            testPercentage = self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()]
            X_train_val_ = []
            y_train_val_ = []
            X_test_ = []
            y_test_ = []

            if self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_RANDOM_FROM_START:
                permIndexes = np.random.permutation(datasetSize).tolist()
                trainIdxs = permIndexes[int(testPercentage * datasetSize):]
                testIdxs = permIndexes[:int(testPercentage * datasetSize)]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            elif self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_RANDOM_FROM_MIDDLE:
                permIndexes = np.random.permutation(datasetSize).tolist()
                midIndex = datasetSize / 2
                sliceSize = testPercentage * datasetSize
                startIndex = int(midIndex - sliceSize / 2)
                endIndex = int(midIndex - sliceSize / 2)

                trainIdxs = permIndexes[:startIndex] + permIndexes[-endIndex:]
                testIdxs = permIndexes[startIndex:-endIndex]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            elif self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_RANDOM_FROM_END:
                permIndexes = np.random.permutation(datasetSize).tolist()
                trainIdxs = permIndexes[:-int(testPercentage * datasetSize)]
                testIdxs = permIndexes[-int(testPercentage * datasetSize):]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            elif self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_FROM_START:
                datasetIndexes = list(range(datasetSize))
                trainIdxs = datasetIndexes[int(testPercentage * datasetSize):]
                testIdxs = datasetIndexes[:int(testPercentage * datasetSize)]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            elif self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_FROM_MIDDLE:
                datasetIndexes = list(range(datasetSize))
                midIndex = datasetSize / 2
                sliceSize = testPercentage * datasetSize
                startIndex = int(midIndex - sliceSize / 2)
                endIndex = int(midIndex - sliceSize / 2)

                trainIdxs = datasetIndexes[:startIndex] + datasetIndexes[-endIndex:]
                testIdxs = datasetIndexes[startIndex:-endIndex]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            elif self.dict_machineLearningParameters[
                self.dkey_mlpTestPercentageDistribution()] == MLPF_PERCENTAGE_DISTRIBUTION_FROM_END:
                datasetIndexes = np.array(list(range(datasetSize)))
                trainIdxs = datasetIndexes[:-int(testPercentage * datasetSize)]
                testIdxs = datasetIndexes[-int(testPercentage * datasetSize):]

                X_train_val_ = np.array(inputArr)[trainIdxs]
                y_train_val_ = np.array(outputArr)[trainIdxs]
                X_test_ = np.array(inputArr)[testIdxs]
                y_test_ = np.array(outputArr)[testIdxs]

            return X_train_val_, y_train_val_, X_test_, y_test_

        # X_full = {}  # Create an input dict to store the values of train_val + test lists
        # y_full = {}  # Create an output dict to store the values of train_val + test lists
        X_train_val = {}  # Create an input dict to store the values of train_val lists
        y_train_val = {}  # Create an output dict to store the values of train_val lists
        X_test = {}  # Create an input dict to store the values of test lists
        y_test = {}  # Create an output dict to store the values of test lists
        # Shorten the name of methodIndex
        methodIndex = self.dict_machineLearningParameters[self.dkey_mlpMethodIndex()]

        inputHeaderColumnsForML = []
        outputHeaderColumnsForML = []

        # Check if the user selected Sequential method
        if self.dict_machineLearningParameters[self.dkey_mlpMethod()] == MLPF_METHOD_SEQUENTIAL:
            for _event_ in dictDataInput.keys():
                # X_full[_event_] = []
                # y_full[_event_] = []
                X_train_val[_event_] = []
                y_train_val[_event_] = []
                X_test[_event_] = []
                y_test[_event_] = []

                tmp_event_arr_input = []
                tmp_event_arr_output = []
                for _index_ in range(0, dictDataInput[_event_].__len__() - (2 * methodIndex + 1)):
                    tmp_arr_input = []
                    tmp_arr_output = []
                    for _index2_ in range(_index_, _index_ + methodIndex):
                        tmp_arr_input.extend(dictDataInput[_event_][_index2_])
                        tmp_arr_output.extend(dictDataOutput[_event_][_index2_ + methodIndex])
                    tmp_event_arr_input.append(tmp_arr_input)
                    tmp_event_arr_output.append(tmp_arr_output)
                    # X_full[_event_].append(tmp_arr_input)
                    # y_full[_event_].append(tmp_arr_output)
                # print("Event: ", _event_)
                # print("InputShape: ", np.array(tmp_event_arr_input).shape)
                # print("OutputShape: ", np.array(tmp_event_arr_output).shape)
                # print()

                X_train_val[_event_], y_train_val[_event_], X_test[_event_], y_test[_event_] = getTrainValTest(
                    tmp_event_arr_input,
                    tmp_event_arr_output)

            for _index_ in range(0, methodIndex):
                for columnName in inputHeaders:
                    inputHeaderColumnsForML.append(columnName + '_SEQ_' + str(_index_))
            for _index_ in range(0, methodIndex):
                for columnName in outputHeaders:
                    outputHeaderColumnsForML.append(columnName + '_SEQ_' + str(_index_))

        # Else if the user selected Average method
        elif self.dict_machineLearningParameters[self.dkey_mlpMethod()] == MLPF_METHOD_AVERAGE:
            for _event_ in dictDataInput.keys():
                # X_full[_event_] = []
                # y_full[_event_] = []
                X_train_val[_event_] = []
                y_train_val[_event_] = []
                X_test[_event_] = []
                y_test[_event_] = []

                tmp_event_arr_input = []
                tmp_event_arr_output = []
                for _index_ in range(0, dictDataInput[_event_].__len__() - (methodIndex + 1)):
                    tmp_arr_input = np.zeros(np.array(dictDataInput[_event_][_index_]).shape)
                    tmp_arr_output = np.zeros(np.array(dictDataOutput[_event_][_index_]).shape)
                    for _index2_ in range(_index_, _index_ + methodIndex):
                        tmp_arr_input += np.array(np.array(dictDataInput[_event_][_index2_]))
                        tmp_arr_output += np.array(np.array(dictDataOutput[_event_][_index2_]))
                    tmp_arr_input /= methodIndex
                    tmp_arr_output /= methodIndex

                    tmp_event_arr_input.append(tmp_arr_input.tolist())
                    tmp_event_arr_output.append(tmp_arr_output.tolist())
                    # X_full[_event_].append(tmp_arr_input.tolist())
                    # y_full[_event_].append(tmp_arr_output.tolist())
                # print("Event: ", _event_)
                # print("InputShape: ", np.array(tmp_event_arr_input).shape)
                # print("OutputShape: ", np.array(tmp_event_arr_output).shape)
                # print()

                X_train_val[_event_], y_train_val[_event_], X_test[_event_], y_test[_event_] = getTrainValTest(
                    tmp_event_arr_input,
                    tmp_event_arr_output)

            inputHeaderColumnsForML = inputHeaders
            outputHeaderColumnsForML = outputHeaders

        # Check if the user selected Sequential Average method
        elif self.dict_machineLearningParameters[self.dkey_mlpMethod()] == MLPF_METHOD_SEQUENTIAL_AVERAGE:
            for _event_ in dictDataInput.keys():
                # X_full[_event_] = []
                # y_full[_event_] = []
                X_train_val[_event_] = []
                y_train_val[_event_] = []
                X_test[_event_] = []
                y_test[_event_] = []

                tmp_event_arr_input = []
                tmp_event_arr_output = []
                for _index_ in range(0, dictDataInput[_event_].__len__() - (2 * methodIndex + 1)):
                    tmp_arr_input = []
                    tmp_arr_output = []

                    tmp_arr_input_mean = np.zeros(np.array(dictDataInput[_event_][_index_]).shape)
                    for _index2_ in range(_index_, _index_ + methodIndex):
                        tmp_arr_input_mean += dictDataInput[_event_][_index2_]
                    tmp_arr_input_mean /= methodIndex

                    for _index2_ in range(_index_, _index_ + methodIndex):
                        # Non absolute values - Uncomment this line for setting the input as non absolute values
                        tmp_input_ext = np.array(dictDataInput[_event_][_index2_]) - tmp_arr_input_mean
                        # Absolute values  - Uncomment this line for setting the input as absolute values
                        # tmp_input_ext = np.absolute(np.array(dictDataInput[_event_][_index2_]) - tmp_arr_input_mean)
                        tmp_arr_input.extend(tmp_input_ext.tolist())
                        tmp_arr_output.extend(dictDataOutput[_event_][_index2_ + methodIndex])

                    tmp_event_arr_input.append(tmp_arr_input)
                    tmp_event_arr_output.append(tmp_arr_output)
                    # X_full[_event_].append(tmp_arr_input)
                    # y_full[_event_].append(tmp_arr_output)
                # print("Event: ", _event_)
                # print("InputShape: ", np.array(tmp_event_arr_input).shape)
                # print("OutputShape: ", np.array(tmp_event_arr_output).shape)
                # print()

                X_train_val[_event_], y_train_val[_event_], X_test[_event_], y_test[_event_] = getTrainValTest(
                    tmp_event_arr_input,
                    tmp_event_arr_output)

            for _index_ in range(0, methodIndex):
                for columnName in inputHeaders:
                    inputHeaderColumnsForML.append(columnName + '_SEQ_' + str(_index_))
            for _index_ in range(0, methodIndex):
                for columnName in outputHeaders:
                    outputHeaderColumnsForML.append(columnName + '_SEQ_' + str(_index_))

        else:
            pass

        return X_train_val, y_train_val, X_test, y_test, inputHeaderColumnsForML, outputHeaderColumnsForML

    # *                                                         * #
    # *********************************************************** #

    def actionButtonExecute(self):
        # FUNCTION FLAGS
        _FF_KEY_DATA = 'Data'
        _FF_KEY_COLUMN_PRIMARY_EVENT_DATA = 'Column Primary Event Data'
        _FF_KEY_PRIMARY_EVENT_UNIQUE_VALUES = 'Primary Event Unique Values'
        _FF_KEY_INPUT_COLUMNS = 'Input Columns'
        _FF_KEY_OUTPUT_COLUMNS = 'Output Columns'
        _FF_KEY_INPUT_COLUMNS_FOR_ML = 'Input Columns Machine Learning'
        _FF_KEY_OUTPUT_COLUMNS_FOR_ML = 'Output Columns Machine Learning'
        _FF_KEY_OUT_COL_HEADER_REAL = 'Output Header Real'
        _FF_KEY_OUT_COL_HEADER_PRED = 'Output Header Predicted'
        _FF_KEY_INP_COL_DENORM_VAL = 'Denormalize Input Values'
        _FF_KEY_OUT_COL_DENORM_VAL = 'Denormalize Output Values'
        _FF_KEY_TRAIN_VAL_ARRAY = 'Training Validation Array'
        _FF_KEY_TEST_ARRAY = 'Test Array'
        _FF_KEY_INPUT = 'Input Array'
        _FF_KEY_OUTPUT = 'Output Array'
        dict_fileData = {}

        # If true run the main pipeline
        if self.dict_tableFilesPaths.keys().__len__() > 0:  # if there is at least a file (safety if)
            # 00 - Error Checking
            for fileName in self.dict_tableFilesPaths.keys():  # for each file in tableFilePaths
                if self.BE_errorExist(fileName):  # if errors exists
                    return  # exit the function

            # 01 - Read the data
            for fileName in self.dict_tableFilesPaths.keys():
                # Create a tmp variable primaryEvent for code simplicity
                primaryEvent = self.dict_tableFilesPaths[fileName][self.dkeyPrimaryEventColumn()]
                dict_fileData[fileName] = {}  # Create a dictionary for file Data
                # read the data and store them in dictionary
                dict_fileData[fileName][_FF_KEY_DATA] = self.BE_readFileData(fileName)

                # Set the input/output columns
                dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS] = self.dict_tableFilesPaths[fileName][
                    self.dkeyInputList()]
                dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS] = self.dict_tableFilesPaths[fileName][
                    self.dkeyOutputList()]

                # Set the normalize/denormalize values
                dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL] = {}
                for _value_ in dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS]:
                    tmp_col_max_value = dict_fileData[fileName][_FF_KEY_DATA][_value_].max()
                    dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL][_value_] = tmp_col_max_value
                dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL] = {}
                for _value_ in dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS]:
                    tmp_col_max_value = dict_fileData[fileName][_FF_KEY_DATA][_value_].max()
                    dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL][_value_] = tmp_col_max_value

                # Set a path for exporting the input-output data

                exportDataFolder = os.path.normpath(self.dict_machineLearningParameters[self.dkey_mlpExportFolder()] +
                                                    '/' + os.path.splitext(fileName)[0] + '/Data')
                file_manip.checkAndCreateFolders(exportDataFolder)  # check if path exists and if not create it
                # Export the input values
                file_manip.exportDictionaryNonList(dictForExport=dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL],
                                                   exportPath=os.path.normpath(
                                                       exportDataFolder + '/InputColumnsDenormValues.csv'),
                                                   headerLine=['InputColumn, DenormValue'])
                # Export the output values
                file_manip.exportDictionaryNonList(dictForExport=dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL],
                                                   exportPath=os.path.normpath(
                                                       exportDataFolder + '/OutputColumnsDenormValues.csv'),
                                                   headerLine=['OutputColumn, DenormValue'])

                # Set the PRIMARY_EVENT_DATA as a dict
                dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA] = {_FF_KEY_INPUT: {}, _FF_KEY_OUTPUT: {}}
                # if primaryEvent value is not None (means the user picked a primary column)
                if primaryEvent is not None:
                    # add the unique event values in the dictionary with key PRIMARY_EVENT_UNIQUE_VALUES
                    dict_fileData[fileName][_FF_KEY_PRIMARY_EVENT_UNIQUE_VALUES] = []
                    [dict_fileData[fileName][_FF_KEY_PRIMARY_EVENT_UNIQUE_VALUES].append(value)
                     for value in dict_fileData[fileName][_FF_KEY_DATA][primaryEvent]
                     if value not in dict_fileData[fileName][_FF_KEY_PRIMARY_EVENT_UNIQUE_VALUES]]

                    # Create a dictionary to normalize the data
                    tmp_dict_normValues = {}
                    for key in dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL].keys():
                        tmp_dict_normValues[key] = dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL][key]
                    for key in dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL].keys():
                        tmp_dict_normValues[key] = dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL][key]

                    # Create the event keys and store the corresponding rows for each key
                    for _event_ in dict_fileData[fileName][_FF_KEY_PRIMARY_EVENT_UNIQUE_VALUES]:
                        tmp_df = dict_fileData[fileName][_FF_KEY_DATA].copy()  # create a tmp copy of the data
                        tmp_df = tmp_df.loc[tmp_df[primaryEvent] == _event_]  # select only the rows of the events
                        del tmp_df[primaryEvent]

                        for key in tmp_df.keys():
                            tmp_df[key] /= tmp_dict_normValues[key]

                        tmp_input_columns = dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS]
                        tmp_output_columns = dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS]

                        dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_INPUT][_event_] = tmp_df[
                            tmp_input_columns].values.tolist()
                        dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_OUTPUT][_event_] = tmp_df[
                            tmp_output_columns].values.tolist()

                else:
                    # Create a dictionary to normalize the date
                    tmp_dict_normValues = {}
                    for key in dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL].keys():
                        tmp_dict_normValues[key] = dict_fileData[fileName][_FF_KEY_INP_COL_DENORM_VAL][key]
                    for key in dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL].keys():
                        tmp_dict_normValues[key] = dict_fileData[fileName][_FF_KEY_OUT_COL_DENORM_VAL][key]

                    tmp_df = dict_fileData[fileName][_FF_KEY_DATA].copy()

                    for key in tmp_df.keys():
                        tmp_df[key] /= tmp_dict_normValues[key]

                    tmp_input_columns = dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS]
                    tmp_output_columns = dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS]

                    dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_INPUT][_FF_KEY_DATA] = tmp_df[
                        tmp_input_columns].values.tolist()
                    dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_OUTPUT][_FF_KEY_DATA] = tmp_df[
                        tmp_output_columns].values.tolist()

                dict_fileData[fileName][_FF_KEY_TRAIN_VAL_ARRAY] = {}
                dict_fileData[fileName][_FF_KEY_TEST_ARRAY] = {}
                dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS_FOR_ML] = []
                dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS_FOR_ML] = []

                (dict_fileData[fileName][_FF_KEY_TRAIN_VAL_ARRAY][_FF_KEY_INPUT],
                 dict_fileData[fileName][_FF_KEY_TRAIN_VAL_ARRAY][_FF_KEY_OUTPUT],
                 dict_fileData[fileName][_FF_KEY_TEST_ARRAY][_FF_KEY_INPUT],
                 dict_fileData[fileName][_FF_KEY_TEST_ARRAY][_FF_KEY_OUTPUT],
                 dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS_FOR_ML],
                 dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS_FOR_ML]) = \
                    self.BE_setTrainValTestArrays(dictDataInput=dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_INPUT],
                                                  dictDataOutput=dict_fileData[fileName][_FF_KEY_COLUMN_PRIMARY_EVENT_DATA][_FF_KEY_OUTPUT],
                                                  inputHeaders=dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS],
                                                  outputHeaders=dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS])

                tmp_input_header_arr = ['Event']
                tmp_input_header_arr.extend(dict_fileData[fileName][_FF_KEY_INPUT_COLUMNS_FOR_ML])
                tmp_output_header_arr = ['Event']
                tmp_output_header_arr.extend(dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS_FOR_ML])

                print("Export INPUT training array...")
                file_manip.exportDictionaryList(
                    dictForExport=dict_fileData[fileName][_FF_KEY_TRAIN_VAL_ARRAY][_FF_KEY_INPUT],
                    exportPath=os.path.normpath(
                        exportDataFolder + '/InputTrainingValidation.csv'),
                    headerLine=tmp_input_header_arr)

                print("Export OUTPUT training array...")
                file_manip.exportDictionaryList(
                    dictForExport=dict_fileData[fileName][_FF_KEY_TRAIN_VAL_ARRAY][_FF_KEY_OUTPUT],
                    exportPath=os.path.normpath(
                        exportDataFolder + '/OutputTrainingValidation.csv'),
                    headerLine=tmp_output_header_arr)

                print("Export INPUT testing array...")
                file_manip.exportDictionaryList(
                    dictForExport=dict_fileData[fileName][_FF_KEY_TEST_ARRAY][_FF_KEY_INPUT],
                    exportPath=os.path.normpath(
                        exportDataFolder + '/InputTest.csv'),
                    headerLine=tmp_input_header_arr)

                print("Export OUTPUT testing array...")
                file_manip.exportDictionaryList(
                    dictForExport=dict_fileData[fileName][_FF_KEY_TEST_ARRAY][_FF_KEY_OUTPUT],
                    exportPath=os.path.normpath(
                        exportDataFolder + '/OutputTest.csv'),
                    headerLine=tmp_output_header_arr)

                # Set the output headers Real/Pred which will be used later in the exported results
                dict_fileData[fileName][_FF_KEY_OUT_COL_HEADER_REAL] = []
                [dict_fileData[fileName][_FF_KEY_OUT_COL_HEADER_REAL].append(col + "_Real")
                 for col in dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS_FOR_ML]]
                dict_fileData[fileName][_FF_KEY_OUT_COL_HEADER_PRED] = []
                [dict_fileData[fileName][_FF_KEY_OUT_COL_HEADER_PRED].append(col + "_Pred")
                 for col in dict_fileData[fileName][_FF_KEY_OUTPUT_COLUMNS_FOR_ML]]

                print(dict_fileData[fileName])

            # 02 - Run Machine Learning Process

    def actionFileListRowChanged_event(self):
        self.listWidget_ColumnList.clear()  # Clear Column Widget
        if self.listWidget_FileList.currentItem() is not None:  # If current item is not None
            self.fileName = self.listWidget_FileList.currentItem().text()  # get current item name
            for column in self.dict_tableFilesPaths[self.fileName][self.dkeyColumnsList()]:  # for each column
                # Add columns as ITEMS to widget
                self.listWidget_ColumnList.addItem(QListWidgetItem(column))

            if self.listWidget_ColumnList.currentItem() is None:  # If COLUMN widget is not None
                self.listWidget_ColumnList.setCurrentRow(0)  # Set first row selected
        else:
            self.fileName = None

    # *********************************************************** #
    # ******************** Helping Functions ******************** #
    # *********************************************************** #
    # *                                                         * #

    def updateInputList(self):
        self.widgetTabInputOutput.listWidget_InputColumns.clear()  # Clear Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][self.dkeyInputList()] is not []:  # if event key is not []
                for event in self.dict_tableFilesPaths[key][self.dkeyInputList()]:  # for each EVENT
                    # Add ITEM to INPUT widget
                    self.widgetTabInputOutput.listWidget_InputColumns.addItem(
                        QListWidgetItem(key + " -> " + event))

    def removeFromInputColumn(self, fileName, column):
        # Remove the specified COLUMN from INPUT_COLUMN_LIST for the specified FILE
        if column in self.dict_tableFilesPaths[fileName][self.dkeyInputList()]:
            self.dict_tableFilesPaths[fileName][self.dkeyInputList()].remove(column)

    def updateOutputList(self):
        self.widgetTabInputOutput.listWidget_OutputColumns.clear()  # Clear Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][self.dkeyOutputList()] is not []:  # if event key is not []
                for event in self.dict_tableFilesPaths[key][self.dkeyOutputList()]:  # for each EVENT
                    # Add ITEM to OUTPUT widget
                    self.widgetTabInputOutput.listWidget_OutputColumns.addItem(
                        QListWidgetItem(key + " -> " + event))

    def removeFromOutputColumn(self, fileName, column):
        # Remove the specified COLUMN from OUTPUT_COLUMN_LIST for the specified FILE
        if column in self.dict_tableFilesPaths[fileName][self.dkeyOutputList()]:
            self.dict_tableFilesPaths[fileName][self.dkeyOutputList()].remove(column)

    def updatePrimaryEventList(self):
        self.widgetTabInputOutput.listWidget_PrimaryEvent.clear()  # Clear Primary Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            # if primary event key is not None
            if self.dict_tableFilesPaths[key][self.dkeyPrimaryEventColumn()] is not None:
                # Add ITEM to PRIMARY_EVENT widget
                self.widgetTabInputOutput.listWidget_PrimaryEvent.addItem(
                    QListWidgetItem(key + " -> " + self.dict_tableFilesPaths[key][self.dkeyPrimaryEventColumn()]))

    def resetPrimEventColumn(self, fileName):
        # Set PRIMARY_COLUMN for the specified FILE to None
        self.dict_tableFilesPaths[fileName][self.dkeyPrimaryEventColumn()] = None

    # *                                                         * #
    # *********************************************************** #

    def actionButtonInput(self):
        # If some file is selected and some columns are selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current columns selected
            currentSelectedItems = self.listWidget_ColumnList.selectedItems()
            for currentColumnSelected in currentSelectedItems:  # for each item selected
                # if this column is not in the INPUT List
                if currentColumnSelected.text() not in self.dict_tableFilesPaths[self.fileName][self.dkeyInputList()]:
                    # Add it to list
                    self.dict_tableFilesPaths[self.fileName][self.dkeyInputList()].append(currentColumnSelected.text())
                    # If this column exist in the private event list
                    if currentColumnSelected.text() == self.dict_tableFilesPaths[self.fileName][
                        self.dkeyPrimaryEventColumn()]:
                        self.resetPrimEventColumn(self.fileName)  # remove it from the list
                        self.updatePrimaryEventList()  # update Input widget
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
                if currentColumnSelected.text() not in self.dict_tableFilesPaths[self.fileName][self.dkeyOutputList()]:
                    # Add it to list
                    self.dict_tableFilesPaths[self.fileName][self.dkeyOutputList()].append(currentColumnSelected.text())
                    # If this column exist in the private event list
                    if currentColumnSelected.text() == self.dict_tableFilesPaths[self.fileName][
                        self.dkeyPrimaryEventColumn()]:
                        self.resetPrimEventColumn(self.fileName)  # remove it from the list
                        self.updatePrimaryEventList()  # update Input widget
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
            if currentColumnSelected in self.dict_tableFilesPaths[self.fileName][self.dkeyInputList()]:
                self.removeFromInputColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateInputList()  # update Input widget
            # If this column exist in the output list
            if currentColumnSelected in self.dict_tableFilesPaths[self.fileName][self.dkeyOutputList()]:
                self.removeFromOutputColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateOutputList()  # update Input widget

            # print(currentFileName, " -> ", currentColumnSelected)

            # Add it to the PRIMARY_EVENT
            self.dict_tableFilesPaths[self.fileName][self.dkeyPrimaryEventColumn()] = currentColumnSelected
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

    # ***** SET SETTINGS GENERAL EVENTS ACTIONS *** #
    def actionExperimentResultChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpExperimentNumber()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.spinBox_ExperimentNumber.value()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpExperimentNumber()])

    def actionTestPercentageChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_TestPercentage.value()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()])

    def actionHoldoutPercentageChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentage()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_HoldoutPercentage.value()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentage()])

    def actionLineEditChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpExportFolder()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.lineEdit_SetOutPath.text()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpExportFolder()])

    def actionButtonSetOutPathClicked(self):
        success, dialog = coFunc.openDirectoryDialog(
            classRef=self,
            dialogName='Choose a Directory',
            dialogOpenAt=self.str_pathToTheProject,
            dialogMultipleSelection=False)

        if success:
            self.widgetTabMachineLearningSettings.tabGeneral.lineEdit_SetOutPath.setText(dialog)

    def actionMachineLearningMethodChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpMethod()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.comboBox_MachineLearningMethods.currentText()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpMethod()])

    def actionMachineLearningMethodIndexChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpMethodIndex()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.spinBox_MachineLearningMethodsIndex.value()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpMethodIndex()])

    def actionMultifileTrainingProcessingChange(self):
        self.dict_machineLearningParameters[self.dkey_multifileTrainingProcessing()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.comboBox_MultifileTrainingProcessing.currentText()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_multifileTrainingProcessing()])

    def actionTestPercentageDistributionChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpTestPercentageDistribution()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.comboBox_TestPercentageDistribution.currentText()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpTestPercentageDistribution()])

    def actionHoldoutPercentageDistributionChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentageDistribution()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.comboBox_HoldoutPercentageDistribution.currentText()
        if self.debugMessageFlag:
            print(self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentageDistribution()])

    # ***** SET SETTINGS MACHINE LEARNING TYPE ACTIONS *** #


# *********************************** #
# *********** Tab Widgets *********** #
# *********************************** #
# *                                 * #

# *********** Machine Learning I/O *********** #
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


# *********** Machine Learning Settings *********** #
class WidgetTabMachineLearningSettings(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # ------------------- #
        # ----- Set Tab ----- #
        # ------------------- #
        self.mainTabWidget = QTabWidget()  # Create a main widget tab

        self.tabGeneral = WidgetTabMachineLearningSettingsGeneral()  # create a tab for General (info)
        self.tabRegressionMethods = WidgetTabMachineLearningSettingsRegressionMethods()

    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        self.tabGeneral.setWidget()  # set tab General (info)
        self.tabRegressionMethods.setWidget()  # set tab Regression methods
        self.mainTabWidget.addTab(self.tabGeneral, "General")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabRegressionMethods, "Regression Methods")  # add tab to mainTabWidget
        self.vbox_main_layout.addWidget(self.mainTabWidget)  # add tabWidget to main layout


# *********** Machine Learning Settings --> General *********** #
class WidgetTabMachineLearningSettingsGeneral(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())  # set the tab style

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # ----------------------- #
        # ----- QPushButton ----- #
        # ----------------------- #
        self.buttonRestoreDefault = QPushButton("Restore Default")
        self.buttonRestoreDefault.setMinimumWidth(150)  # Set Minimum Width
        self.buttonRestoreDefault.setMinimumHeight(30)  # Set Minimum Height
        self.buttonRestoreDefault.setToolTip('Restore the default values.')  # Add Description

        self.buttonSetOutPath = QPushButton("...")
        self.buttonSetOutPath.setMinimumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonSetOutPath.setMaximumWidth(INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonSetOutPath.setMinimumHeight(30)  # Set Minimum Height
        self.buttonSetOutPath.setToolTip('Select the output path')  # Add Description

        # -------------------- #
        # ----- QSpinBox ----- #
        # -------------------- #
        # create a spinBox for the ExperimentNumber
        self.spinBox_ExperimentNumber = QSpinBox()
        # set the minimum value to 1 (at least an experiments must be performed)
        self.spinBox_ExperimentNumber.setMinimum(1)

        # create a spinBox for the MachineLearningMethodIndex
        self.spinBox_MachineLearningMethodsIndex = QSpinBox()
        # set the minimum value to 1 (at least a row needs to be used as input/output)
        self.spinBox_MachineLearningMethodsIndex.setMinimum(1)

        # -------------------------- #
        # ----- QDoubleSpinBox ----- #
        # -------------------------- #
        # create a doubleSpinBox for the test percentage
        self.doubleSpinBox_TestPercentage = QDoubleSpinBox()
        # set minimum value
        self.doubleSpinBox_TestPercentage.setMinimum(MLF_TEST_PERCENTAGE_MIN)
        # set maximum value
        self.doubleSpinBox_TestPercentage.setMaximum(MLF_TEST_PERCENTAGE_MAX)
        # set step
        self.doubleSpinBox_TestPercentage.setSingleStep(MLF_TEST_STEP_WIDGET)

        self.doubleSpinBox_HoldoutPercentage = QDoubleSpinBox()
        # set minimum value
        self.doubleSpinBox_HoldoutPercentage.setMinimum(MLF_HOLDOUT_PERCENTAGE_MIN)
        # set maximum value
        self.doubleSpinBox_HoldoutPercentage.setMaximum(MLF_HOLDOUT_PERCENTAGE_MAX)
        # set step
        self.doubleSpinBox_HoldoutPercentage.setSingleStep(MLF_HOLDOUT_STEP_WIDGET)

        # --------------------- #
        # ----- QLineEdit ----- #
        # --------------------- #
        self.lineEdit_SetOutPath = QLineEdit()
        self.lineEdit_SetOutPath.setEnabled(False)

        # --------------------- #
        # ----- QComboBox ----- #
        # --------------------- #
        # MachineLearningMethods
        self.comboBox_MachineLearningMethods = QComboBox()
        self.comboBox_MachineLearningMethods.setMinimumWidth(150)
        self.comboBox_MachineLearningMethods.addItems(MLPF_METHOD_LIST)

        # MultifileTrainingProcessing
        self.comboBox_MultifileTrainingProcessing = QComboBox()
        self.comboBox_MultifileTrainingProcessing.setMinimumWidth(200)
        self.comboBox_MultifileTrainingProcessing.addItems(MLPF_MULTIFILE_TRAINING_PROCESSING_LIST)

        # TestPercentageDistribution
        self.comboBox_TestPercentageDistribution = QComboBox()
        self.comboBox_TestPercentageDistribution.setMinimumWidth(200)
        self.comboBox_TestPercentageDistribution.addItems(MLPF_PERCENTAGE_DISTRIBUTION_LIST)

        # HoldoutPercentageDistribution
        self.comboBox_HoldoutPercentageDistribution = QComboBox()
        self.comboBox_HoldoutPercentageDistribution.setMinimumWidth(200)
        self.comboBox_HoldoutPercentageDistribution.addItems(MLPF_PERCENTAGE_DISTRIBUTION_LIST)

        # ------------------------------ #
        # ----- Set Default Values ----- #
        # ------------------------------ #
        self._experimentNumberDefaultValue = MLF_DEFAULT_EXPERIMENT_VALUE
        self._testPercentageDefaultValue = MLF_DEFAULT_TEST_PERCENTAGE
        self._holdoutPercentageDefaultValue = MLF_DEFAULT_HOLDOUT_PERCENTAGE
        self._exportPathDefaultValue = MLF_DEFAULT_EXPORT_FOLDER_PATH
        self._mlMethodDefaultValue = MLF_DEFAULT_METHOD
        self._mlMethodIndexDefaultValue = MLF_DEFAULT_METHOD_INDEX
        self._mlMultifileTrainingProcessingDefaultValue = MLF_DEFAULT_MULTIFILE_TRAINING_PROCESSING
        self._testPercentageDistributionDefaultValue = MLF_DEFAULT_TEST_PERCENTAGE_DISTRIBUTION
        self._holdoutPercentageDistributionDefaultValue = MLF_DEFAULT_HOLDOUT_PERCENTAGE_DISTRIBUTION

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        self.restoreDefaultValues()
        self.setEvents_()

        # Set Label
        label_ExperimentalNumber = QLabel("Number of Experiments:")
        # label_ExperimentalNumber.setMinimumWidth(100)
        label_TestPercentage = QLabel("Test Percentage (0.00-0.75):")
        # label_TestPercentage.setMinimumWidth(100)
        label_TestPercentageDistribution = QLabel("Test Percentage Distribution:")
        # label_TestPercentageDistribution.setMinimumWidth(100)
        label_HoldOutPercentage = QLabel("Hold-Out Percentage (0.00-0.75):")
        # label_HoldOutPercentage.setMinimumWidth(100)
        label_HoldoutPercentageDistribution = QLabel("Hold-Out Percentage Distribution:")
        # label_HoldoutPercentageDistribution.setMinimumWidth(100)
        label_SetOutPath = QLabel("Export Folder:")
        # label_SetOutPath.setMinimumWidth(100)
        label_MachineLearningMethod = QLabel("Machine Learning Method:")
        # label_MachineLearningMethod.setMinimumWidth(100)
        label_MachineLearningMethodIndex = QLabel("Method Usage Index:")
        # label_MachineLearningMethodIndex.setMinimumWidth(100)
        label_MultifileTrainingProcessing = QLabel("Multifile Training Processing:")
        # label_MultifileTrainingProcessing.setMinimumWidth(100)

        # Set SpinBoxes
        hbox_ExperimentalNumber = QHBoxLayout()
        hbox_ExperimentalNumber.addWidget(label_ExperimentalNumber)
        hbox_ExperimentalNumber.addWidget(self.spinBox_ExperimentNumber)
        hbox_ExperimentalNumber.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TestPercentage = QHBoxLayout()
        hbox_TestPercentage.addWidget(label_TestPercentage)
        hbox_TestPercentage.addWidget(self.doubleSpinBox_TestPercentage)
        hbox_TestPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TestPercentageDistribution = QHBoxLayout()
        hbox_TestPercentageDistribution.addWidget(label_TestPercentageDistribution)
        hbox_TestPercentageDistribution.addWidget(self.comboBox_TestPercentageDistribution)
        hbox_TestPercentageDistribution.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_HoldOutPercentage = QHBoxLayout()
        hbox_HoldOutPercentage.addWidget(label_HoldOutPercentage)
        hbox_HoldOutPercentage.addWidget(self.doubleSpinBox_HoldoutPercentage)
        hbox_HoldOutPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_HoldOutPercentageDistribution = QHBoxLayout()
        hbox_HoldOutPercentageDistribution.addWidget(label_HoldoutPercentageDistribution)
        hbox_HoldOutPercentageDistribution.addWidget(self.comboBox_HoldoutPercentageDistribution)
        hbox_HoldOutPercentageDistribution.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_finalSpinBoxes = QVBoxLayout()
        vbox_finalSpinBoxes.addLayout(hbox_ExperimentalNumber)
        vbox_finalSpinBoxes.addLayout(hbox_TestPercentage)
        vbox_finalSpinBoxes.addLayout(hbox_TestPercentageDistribution)
        vbox_finalSpinBoxes.addLayout(hbox_HoldOutPercentage)
        vbox_finalSpinBoxes.addLayout(hbox_HoldOutPercentageDistribution)

        # SetOutPath Layout
        hbox_SetOutPath = QHBoxLayout()
        hbox_SetOutPath.addWidget(label_SetOutPath)
        hbox_SetOutPath.addWidget(self.lineEdit_SetOutPath)
        hbox_SetOutPath.addWidget(self.buttonSetOutPath)
        # hbox_SetOutPath.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        # Button Layout
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))
        hbox_buttons.addWidget(self.buttonRestoreDefault)

        # MachineLearningMethods
        hbox_MachineLearningMethods = QHBoxLayout()
        hbox_MachineLearningMethods.addWidget(label_MachineLearningMethod)
        hbox_MachineLearningMethods.addWidget(self.comboBox_MachineLearningMethods)
        hbox_MachineLearningMethods.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_MachineLearningMethodsIndex = QHBoxLayout()
        hbox_MachineLearningMethodsIndex.addWidget(label_MachineLearningMethodIndex)
        hbox_MachineLearningMethodsIndex.addWidget(self.spinBox_MachineLearningMethodsIndex)
        hbox_MachineLearningMethodsIndex.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_finalMachineLearningMethods = QVBoxLayout()
        vbox_finalMachineLearningMethods.addLayout(hbox_MachineLearningMethods)
        vbox_finalMachineLearningMethods.addLayout(hbox_MachineLearningMethodsIndex)

        # MultifileTrainingProcessing
        hbox_MultifileTrainingProcessing = QHBoxLayout()
        hbox_MultifileTrainingProcessing.addWidget(label_MultifileTrainingProcessing)
        hbox_MultifileTrainingProcessing.addWidget(self.comboBox_MultifileTrainingProcessing)
        hbox_MultifileTrainingProcessing.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        self.vbox_main_layout.addLayout(vbox_finalSpinBoxes)
        self.vbox_main_layout.addLayout(hbox_SetOutPath)
        self.vbox_main_layout.addLayout(vbox_finalMachineLearningMethods)
        self.vbox_main_layout.addLayout(hbox_MultifileTrainingProcessing)
        # self.vbox_main_layout.addSpacerItem(QSpacerItem(0, INT_MAX_STRETCH))
        self.vbox_main_layout.addLayout(hbox_buttons)

    def restoreDefaultValues(self):
        # set default value
        self.spinBox_ExperimentNumber.setValue(self._experimentNumberDefaultValue)
        self.doubleSpinBox_TestPercentage.setValue(self._testPercentageDefaultValue)
        self.doubleSpinBox_HoldoutPercentage.setValue(self._holdoutPercentageDefaultValue)
        self.lineEdit_SetOutPath.setText(os.path.normpath(self._exportPathDefaultValue))
        self.comboBox_MachineLearningMethods.setCurrentText(self._mlMethodDefaultValue)
        self.spinBox_MachineLearningMethodsIndex.setValue(self._mlMethodIndexDefaultValue)
        self.comboBox_MultifileTrainingProcessing.setCurrentText(self._mlMultifileTrainingProcessingDefaultValue)
        self.comboBox_TestPercentageDistribution.setCurrentText(self._testPercentageDistributionDefaultValue)
        self.comboBox_HoldoutPercentageDistribution.setCurrentText(self._holdoutPercentageDistributionDefaultValue)

    def setEvents_(self):
        self.buttonRestoreDefault.clicked.connect(self.restoreDefaultValues)

    # ------------------------------ #
    # ----- GET DEFAULT VALUES ----- #
    # ------------------------------ #
    def getDefaultExperimentNumber(self):
        return self._experimentNumberDefaultValue

    def getDefaultTestPercentage(self):
        return self._testPercentageDefaultValue

    def getDefaultTestPercentageDistribution(self):
        return self._testPercentageDistributionDefaultValue

    def getDefaultHoldoutPercentage(self):
        return self._holdoutPercentageDefaultValue

    def getDefaultHoldoutPercentageDistribution(self):
        return self._holdoutPercentageDistributionDefaultValue

    def getDefaultExportPath(self):
        return self._exportPathDefaultValue

    def getDefaultMethod(self):
        return self._mlMethodDefaultValue

    def getDefaultMethodIndex(self):
        return self._mlMethodIndexDefaultValue

    def getDefaultMultifileTrainingProcessing(self):
        return self._mlMultifileTrainingProcessingDefaultValue


# *********** Machine Learning Settings --> Regression Methods *********** #
class WidgetTabMachineLearningSettingsRegressionMethods(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())  # set the tab style

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # ----------------------- #
        # ----- PushButtons ----- #
        # ----------------------- #
        _icon = QIcon(QPixmap(ICON_OPTION_SETTINGS))
        self.button_Ridge = QPushButton()
        self.button_BayesianRidge = QPushButton()
        self.button_Lasso = QPushButton()
        self.button_LassoLars = QPushButton()
        self.button_TweedieRegressor = QPushButton()
        self.button_SGDRegressor = QPushButton()
        self.button_SVR = QPushButton()
        self.button_LinearSVR = QPushButton()
        self.button_SVR = QPushButton()
        self.button_NearestNeighbor = QPushButton()
        self.button_KNeighborsRegressor = QPushButton()
        self.button_DecisionTreeRegressor = QPushButton()
        self.button_RandomForestRegressor = QPushButton()
        self.button_AdaBoostRegressor = QPushButton()
        self.button_GradientBoostingRegressor = QPushButton()

        self.button_Ridge.setIcon(_icon)
        self.button_BayesianRidge.setIcon(_icon)
        self.button_Lasso.setIcon(_icon)
        self.button_LassoLars.setIcon(_icon)
        self.button_TweedieRegressor.setIcon(_icon)
        self.button_SGDRegressor.setIcon(_icon)
        self.button_SVR.setIcon(_icon)
        self.button_LinearSVR.setIcon(_icon)
        self.button_SVR.setIcon(_icon)
        self.button_NearestNeighbor.setIcon(_icon)
        self.button_KNeighborsRegressor.setIcon(_icon)
        self.button_DecisionTreeRegressor.setIcon(_icon)
        self.button_RandomForestRegressor.setIcon(_icon)
        self.button_AdaBoostRegressor.setIcon(_icon)
        self.button_GradientBoostingRegressor.setIcon(_icon)

        # ---------------------- #
        # ----- CheckBoxes ----- #
        # ---------------------- #
        self.checkbox_LinearRegression = QCheckBox()
        self.checkbox_Ridge = QCheckBox()
        self.checkbox_BayesianRidge = QCheckBox()
        self.checkbox_Lasso = QCheckBox()
        self.checkbox_LassoLars = QCheckBox()
        self.checkbox_TweedieRegressor = QCheckBox()
        self.checkbox_SGDRegressor = QCheckBox()
        self.checkbox_SVR = QCheckBox()
        self.checkbox_LinearSVR = QCheckBox()
        self.checkbox_SVR = QCheckBox()
        self.checkbox_NearestNeighbor = QCheckBox()
        self.checkbox_KNeighborsRegressor = QCheckBox()
        self.checkbox_DecisionTreeRegressor = QCheckBox()
        self.checkbox_RandomForestRegressor = QCheckBox()
        self.checkbox_AdaBoostRegressor = QCheckBox()
        self.checkbox_GradientBoostingRegressor = QCheckBox()

        # ---------------------- #
        # ----- ScrollArea ----- #
        # ---------------------- #
        self.scrollArea_regMethods = QScrollArea()

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        self.scrollArea_regMethods.setWidgetResizable(True)
        self.scrollArea_regMethods.setWidget(self._setGridLayout())
        self.vbox_main_layout.addWidget(self.scrollArea_regMethods)

    def _setGridLayout(self):
        # label_min_width = 200

        # Set Label
        label_Method = QLabel('<b><u>Method<\\u><\\b>')
        # label_Method.setMaximumHeight(30)

        label_State = QLabel('<b><u>State<\\u><\\b>')
        # label_State.setMaximumHeight(30)

        label_Options = QLabel('<b><u>Options<\\u><\\b>')
        # label_Options.setMaximumHeight(30)

        label_LinearRegression = QLabel(mlr.ML_REG_LINEAR_REGRESSION)
        # label_LinearRegression.setMinimumWidth(label_min_width)
        label_Ridge = QLabel(mlr.ML_REG_RIDGE)
        # label_Ridge.setMinimumWidth(label_min_width)
        label_BayesianRidge = QLabel(mlr.ML_REG_BAYESIAN_RIDGE)
        # label_BayesianRidge.setMinimumWidth(label_min_width)
        label_Lasso = QLabel(mlr.ML_REG_LASSO)
        # label_Lasso.setMinimumWidth(label_min_width)
        label_LassoLars = QLabel(mlr.ML_REG_LASSO_LARS)
        # label_LassoLars.setMinimumWidth(label_min_width)
        label_TweedieRegressor = QLabel(mlr.ML_REG_TWEEDIE_REGRESSOR)
        # label_TweedieRegressor.setMinimumWidth(label_min_width)
        label_SGDRegressor = QLabel(mlr.ML_REG_SGD_REGRESSOR)
        # label_SGDRegressor.setMinimumWidth(label_min_width)
        label_SVR = QLabel(mlr.ML_REG_SVR)
        # label_SVR.setMinimumWidth(label_min_width)
        label_LinearSVR = QLabel(mlr.ML_REG_LINEAR_SVR)
        # label_LinearSVR.setMinimumWidth(label_min_width)
        label_NearestNeighbor = QLabel(mlr.ML_REG_NEAREST_NEIGHBORS)
        # label_NearestNeighbor.setMinimumWidth(label_min_width)
        label_KNeighborsRegressor = QLabel(mlr.ML_REG_K_NEIGHBORS_REGRESSOR)
        # label_KNeighborsRegressor.setMinimumWidth(label_min_width)
        label_DecisionTreeRegressor = QLabel(mlr.ML_REG_DECISION_TREE_REGRESSOR)
        # label_DecisionTreeRegressor.setMinimumWidth(label_min_width)
        label_RandomForestRegressor = QLabel(mlr.ML_REG_RANDOM_FOREST_REGRESSOR)
        # label_RandomForestRegressor.setMinimumWidth(label_min_width)
        label_AdaBoostRegressor = QLabel(mlr.ML_REG_ADA_BOOST_REGRESSOR)
        # label_AdaBoostRegressor.setMinimumWidth(label_min_width)
        label_GradientBoostingRegressor = QLabel(mlr.ML_REG_GRADIENT_BOOSTING_REGRESSOR)
        # label_GradientBoostingRegressor.setMinimumWidth(label_min_width)

        # Set layout
        scrollAreaWidget = QWidget()
        scrollAreaWidget.setMaximumWidth(840)
        scrollAreaWidget.setMaximumHeight(512)
        gridBox_Methods = QGridLayout(scrollAreaWidget)

        gridBox_Methods.addWidget(label_Method, 0, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(label_State, 0, 1, alignment=Qt.AlignCenter)
        gridBox_Methods.addWidget(label_Options, 0, 2, alignment=Qt.AlignCenter)

        gridBox_Methods.addWidget(label_LinearRegression, 1, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_LinearRegression, 1, 1, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_Ridge, 2, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_Ridge, 2, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_Ridge, 2, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_BayesianRidge, 3, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_BayesianRidge, 3, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_BayesianRidge, 3, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_Lasso, 4, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_Lasso, 4, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_Lasso, 4, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_LassoLars, 5, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_LassoLars, 5, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_LassoLars, 5, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_TweedieRegressor, 6, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_TweedieRegressor, 6, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_TweedieRegressor, 6, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_SGDRegressor, 7, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_SGDRegressor, 7, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_SGDRegressor, 7, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_SVR, 8, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_SVR, 8, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_SVR, 8, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_LinearSVR, 9, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_LinearSVR, 9, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_LinearSVR, 9, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_NearestNeighbor, 10, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_NearestNeighbor, 10, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_NearestNeighbor, 10, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_KNeighborsRegressor, 11, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_KNeighborsRegressor, 11, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_KNeighborsRegressor, 11, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_DecisionTreeRegressor, 12, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_DecisionTreeRegressor, 12, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_DecisionTreeRegressor, 12, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_RandomForestRegressor, 13, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_RandomForestRegressor, 13, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_RandomForestRegressor, 13, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_AdaBoostRegressor, 14, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_AdaBoostRegressor, 14, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_AdaBoostRegressor, 14, 2, alignment=Qt.AlignHCenter)

        gridBox_Methods.addWidget(label_GradientBoostingRegressor, 15, 0, alignment=Qt.AlignLeft)
        gridBox_Methods.addWidget(self.checkbox_GradientBoostingRegressor, 15, 1, alignment=Qt.AlignHCenter)
        gridBox_Methods.addWidget(self.button_GradientBoostingRegressor, 15, 2, alignment=Qt.AlignHCenter)

        return scrollAreaWidget

    def _setHorLayout(self):
        label_min_width = 200

        # Set Label
        label_Method = QLabel('Method')
        label_Method.setMinimumWidth(label_min_width)

        label_LinearRegression = QLabel(mlr.ML_REG_LINEAR_REGRESSION)
        label_LinearRegression.setMinimumWidth(label_min_width)
        label_Ridge = QLabel(mlr.ML_REG_RIDGE)
        label_Ridge.setMinimumWidth(label_min_width)
        label_BayesianRidge = QLabel(mlr.ML_REG_BAYESIAN_RIDGE)
        label_BayesianRidge.setMinimumWidth(label_min_width)
        label_Lasso = QLabel(mlr.ML_REG_LASSO)
        label_Lasso.setMinimumWidth(label_min_width)
        label_LassoLars = QLabel(mlr.ML_REG_LASSO_LARS)
        label_LassoLars.setMinimumWidth(label_min_width)
        label_TweedieRegressor = QLabel(mlr.ML_REG_TWEEDIE_REGRESSOR)
        label_TweedieRegressor.setMinimumWidth(label_min_width)
        label_SGDRegressor = QLabel(mlr.ML_REG_SGD_REGRESSOR)
        label_SGDRegressor.setMinimumWidth(label_min_width)
        label_SVR = QLabel(mlr.ML_REG_SVR)
        label_SVR.setMinimumWidth(label_min_width)
        label_LinearSVR = QLabel(mlr.ML_REG_LINEAR_SVR)
        label_LinearSVR.setMinimumWidth(label_min_width)
        label_NearestNeighbor = QLabel(mlr.ML_REG_NEAREST_NEIGHBORS)
        label_NearestNeighbor.setMinimumWidth(label_min_width)
        label_KNeighborsRegressor = QLabel(mlr.ML_REG_K_NEIGHBORS_REGRESSOR)
        label_KNeighborsRegressor.setMinimumWidth(label_min_width)
        label_DecisionTreeRegressor = QLabel(mlr.ML_REG_DECISION_TREE_REGRESSOR)
        label_DecisionTreeRegressor.setMinimumWidth(label_min_width)
        label_RandomForestRegressor = QLabel(mlr.ML_REG_RANDOM_FOREST_REGRESSOR)
        label_RandomForestRegressor.setMinimumWidth(label_min_width)
        label_AdaBoostRegressor = QLabel(mlr.ML_REG_ADA_BOOST_REGRESSOR)
        label_AdaBoostRegressor.setMinimumWidth(label_min_width)
        label_GradientBoostingRegressor = QLabel(mlr.ML_REG_GRADIENT_BOOSTING_REGRESSOR)
        label_GradientBoostingRegressor.setMinimumWidth(label_min_width)

        # Set hboxes
        hbox_header = QHBoxLayout()
        hbox_LinearRegression = QHBoxLayout()
        hbox_Ridge = QHBoxLayout()
        hbox_BayesianRidge = QHBoxLayout()
        hbox_Lasso = QHBoxLayout()
        hbox_LassoLars = QHBoxLayout()
        hbox_TweedieRegressor = QHBoxLayout()
        hbox_SGDRegressor = QHBoxLayout()
        hbox_SVR = QHBoxLayout()
        hbox_LinearSVR = QHBoxLayout()
        hbox_NearestNeighbor = QHBoxLayout()
        hbox_KNeighborsRegressor = QHBoxLayout()
        hbox_DecisionTreeRegressor = QHBoxLayout()
        hbox_RandomForestRegressor = QHBoxLayout()
        hbox_AdaBoostRegressor = QHBoxLayout()
        hbox_GradientBoostingRegressor = QHBoxLayout()

        # Add to hboxes
        hbox_header.addWidget(label_Method)
        hbox_header.addWidget(QLabel("State"))
        hbox_header.addWidget(QLabel("Options"))

        hbox_LinearRegression.addWidget(label_LinearRegression)
        hbox_LinearRegression.addWidget(self.checkbox_LinearRegression)
        hbox_LinearRegression.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_Ridge.addWidget(label_Ridge)
        hbox_Ridge.addWidget(self.checkbox_Ridge)
        hbox_Ridge.addWidget(self.button_Ridge)
        hbox_Ridge.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_BayesianRidge.addWidget(label_BayesianRidge)
        hbox_BayesianRidge.addWidget(self.checkbox_BayesianRidge)
        hbox_BayesianRidge.addWidget(self.button_BayesianRidge)
        hbox_BayesianRidge.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_Lasso.addWidget(label_Lasso)
        hbox_Lasso.addWidget(self.checkbox_Lasso)
        hbox_Lasso.addWidget(self.button_Lasso)
        hbox_Lasso.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_LassoLars.addWidget(label_LassoLars)
        hbox_LassoLars.addWidget(self.checkbox_LassoLars)
        hbox_LassoLars.addWidget(self.button_LassoLars)
        hbox_LassoLars.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TweedieRegressor.addWidget(label_TweedieRegressor)
        hbox_TweedieRegressor.addWidget(self.checkbox_TweedieRegressor)
        hbox_TweedieRegressor.addWidget(self.button_TweedieRegressor)
        hbox_TweedieRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_SGDRegressor.addWidget(label_SGDRegressor)
        hbox_SGDRegressor.addWidget(self.checkbox_SGDRegressor)
        hbox_SGDRegressor.addWidget(self.button_SGDRegressor)
        hbox_SGDRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_SVR.addWidget(label_SVR)
        hbox_SVR.addWidget(self.checkbox_SVR)
        hbox_SVR.addWidget(self.button_SVR)
        hbox_SVR.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_LinearSVR.addWidget(label_LinearSVR)
        hbox_LinearSVR.addWidget(self.checkbox_LinearSVR)
        hbox_LinearSVR.addWidget(self.button_LinearSVR)
        hbox_LinearSVR.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_NearestNeighbor.addWidget(label_NearestNeighbor)
        hbox_NearestNeighbor.addWidget(self.checkbox_NearestNeighbor)
        hbox_NearestNeighbor.addWidget(self.button_NearestNeighbor)
        hbox_NearestNeighbor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_KNeighborsRegressor.addWidget(label_KNeighborsRegressor)
        hbox_KNeighborsRegressor.addWidget(self.checkbox_KNeighborsRegressor)
        hbox_KNeighborsRegressor.addWidget(self.button_KNeighborsRegressor)
        hbox_KNeighborsRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_DecisionTreeRegressor.addWidget(label_DecisionTreeRegressor)
        hbox_DecisionTreeRegressor.addWidget(self.checkbox_DecisionTreeRegressor)
        hbox_DecisionTreeRegressor.addWidget(self.button_DecisionTreeRegressor)
        hbox_DecisionTreeRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_RandomForestRegressor.addWidget(label_RandomForestRegressor)
        hbox_RandomForestRegressor.addWidget(self.checkbox_RandomForestRegressor)
        hbox_RandomForestRegressor.addWidget(self.button_RandomForestRegressor)
        hbox_RandomForestRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_AdaBoostRegressor.addWidget(label_AdaBoostRegressor)
        hbox_AdaBoostRegressor.addWidget(self.checkbox_AdaBoostRegressor)
        hbox_AdaBoostRegressor.addWidget(self.button_AdaBoostRegressor)
        hbox_AdaBoostRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_GradientBoostingRegressor.addWidget(label_GradientBoostingRegressor)
        hbox_GradientBoostingRegressor.addWidget(self.checkbox_GradientBoostingRegressor)
        hbox_GradientBoostingRegressor.addWidget(self.button_GradientBoostingRegressor)
        hbox_GradientBoostingRegressor.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_Methods = QVBoxLayout()
        vbox_Methods.addLayout(hbox_header)
        vbox_Methods.addLayout(hbox_LinearRegression)
        vbox_Methods.addLayout(hbox_Ridge)
        vbox_Methods.addLayout(hbox_BayesianRidge)
        vbox_Methods.addLayout(hbox_Lasso)
        vbox_Methods.addLayout(hbox_LassoLars)
        vbox_Methods.addLayout(hbox_TweedieRegressor)
        vbox_Methods.addLayout(hbox_SGDRegressor)
        vbox_Methods.addLayout(hbox_SVR)
        vbox_Methods.addLayout(hbox_LinearSVR)
        vbox_Methods.addLayout(hbox_NearestNeighbor)
        vbox_Methods.addLayout(hbox_KNeighborsRegressor)
        vbox_Methods.addLayout(hbox_DecisionTreeRegressor)
        vbox_Methods.addLayout(hbox_RandomForestRegressor)
        vbox_Methods.addLayout(hbox_AdaBoostRegressor)
        vbox_Methods.addLayout(hbox_GradientBoostingRegressor)

        return vbox_Methods

# *                                 * #
# *********************************** #

# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetMachineLearningMainWidget(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                                winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')