import os.path
import sys
import matplotlib

# from PySide2.QtCore import (
#     QUrl
# )

from PySide2.QtWidgets import (
    QWidget,
    QApplication,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
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
    QComboBox
)

from PySide2.QtGui import (
    QIcon,
    QPixmap
)

import lib.core.my_calendar_v2 as my_cal_v2
from lib.core.project_flags import *

from lib.gui.guiStyle import setStyle_
import lib.gui.commonFunctions as coFunc

# *************************************************************************************************** #

matplotlib.use("Agg")  # Set matplotlib to use non-interface (don't plot figures)


class WidgetMachineLearningMainWidget(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=None, maxH=None,
                 winTitle='My Window', iconPath=''):
        super().__init__()
        self.setStyleSheet(setStyle_())  # Set the styleSheet

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
        self._DKEY_MLP_VALIDATION_PERCENTAGE: str = 'validation-percentage'
        self._DKEY_MLP_EXPORT_FOLDER: str = 'export-folder'
        self._DKEY_MLP_HOLDOUT_SIZE: str = 'holdout-size'
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
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
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
            self.dkey_mlpHoldoutPercentage(): self.widgetTabMachineLearningSettings.tabGeneral.getDefaultHoldoutPercentage(),
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

    def dkey_mlpValidationPercentage(self):
        return self._DKEY_MLP_VALIDATION_PERCENTAGE

    def dkey_mlpExportFolder(self):
        return self._DKEY_MLP_EXPORT_FOLDER

    def dkey_mlpHoldoutPercentage(self):
        return self._DKEY_MLP_HOLDOUT_SIZE

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

    def addItemsToList(self, fullPath, splitter=my_cal_v2.del_comma):
        fileName = fullPath.split('/')[-1:][0]  # find the name of the file
        # Create the dictionary
        self.dict_tableFilesPaths[fileName] = {self.dkeyFileName(): fileName,
                                               self.dkeyFullPath(): fullPath,
                                               self.dkeyColumnsList(): file_manip.getColumnNames(fullPath,
                                                                                                 splitter=splitter),
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
        # Open a dialog for CSV files
        success, dialog = coFunc.openFileDialog(
            classRef=self,
            dialogName='Open Table File (Currently strictly CSV)',
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
    # *****(***** Helping Functions for ButtonExecute *********** #
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

    def BE_linearExecution(self):
        pass

    def BE_parallelExecution(self):
        pass

    # *                                                         * #
    # *********************************************************** #

    def actionButtonExecute(self):
        pass

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
                    if currentColumnSelected.text() == self.dict_tableFilesPaths[self.fileName][self.dkeyPrimaryEventColumn()]:
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
                    if currentColumnSelected.text() == self.dict_tableFilesPaths[self.fileName][self.dkeyPrimaryEventColumn()]:
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
        # print(self.dict_machineLearningParameters[self.dkey_mlpExperimentNumber()])

    def actionTestPercentageChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_TestPercentage.value()
        # print(self.dict_machineLearningParameters[self.dkey_mlpTestPercentage()])

    def actionHoldoutPercentageChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentage()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.doubleSpinBox_HoldoutPercentage.value()
        # print(self.dict_machineLearningParameters[self.dkey_mlpHoldoutPercentage()])

    def actionLineEditChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpExportFolder()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.lineEdit_SetOutPath.text()
        # print(self.dict_machineLearningParameters[self.dkey_mlpExportFolder()])

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
        # print(self.dict_machineLearningParameters[self.dkey_mlpMethod()])

    def actionMachineLearningMethodIndexChange(self):
        self.dict_machineLearningParameters[self.dkey_mlpMethodIndex()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.spinBox_MachineLearningMethodsIndex.value()
        # print(self.dict_machineLearningParameters[self.dkey_mlpMethodIndex()])

    def actionMultifileTrainingProcessingChange(self):
        self.dict_machineLearningParameters[self.dkey_multifileTrainingProcessing()] = \
            self.widgetTabMachineLearningSettings.tabGeneral.comboBox_MultifileTrainingProcessing.currentText()
        # print(self.dict_machineLearningParameters[self.dkey_multifileTrainingProcessing()])

    # ***** SET SETTINGS MACHINE LEARNING TYPE ACTIONS *** #

    # ***** SET SETTINGS LINEAR REGRESSION EVENTS ACTIONS *** #

    # ***** SET SETTINGS RIDGE EVENTS ACTIONS *** #

    # ***** SET SETTINGS LASSO EVENTS ACTIONS *** #

    # ***** SET SETTINGS DECISION TREE REGRESSOR EVENTS ACTIONS *** #

    # ***** SET SETTINGS GRADIENT BOOSTING REGRESSOR EVENTS ACTIONS *** #

    # ***** SET SETTINGS ADA BOOST REGRESSOR EVENTS ACTIONS *** #

    # ***** SET SETTINGS K-NEIGHBORS REGRESSOR EVENTS ACTIONS *** #

    # ------------------------- #
    # ----- Message Boxes ----- #
    # ------------------------- #


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
        self.tabLinearRegression = WidgetTabMachineLearningSettingsLinearRegression()  # create a tab for LinearRegression
        self.tabRidge = WidgetTabMachineLearningSettingsRidge()  # create a tab for Ridge
        self.tabLasso = WidgetTabMachineLearningSettingsLasso()  # create a tab for Lasso
        self.tabDecisionTreeRegressor = WidgetTabMachineLearningSettingsDecisionTreeRegressor()  # create a tab for DecisionTreeRegressor
        self.tabRandomForestRegressor = WidgetTabMachineLearningSettingsRandomForestRegressor()  # create a tab for RandomForestRegressor
        self.tabGradientBoostingRegressor = WidgetTabMachineLearningSettingsGradientBoostingRegressor()  # create a tab for GradientBoostingRegressor
        self.tabAdaBoostRegressor = WidgetTabMachineLearningSettingsAdaBoostRegressor()  # create a tab for AdaBoostRegressor
        self.tabKNeighborsRegressor = WidgetTabMachineLearningSettingsKNeighborsRegressor()  # create a tab for KNeighborsRegressor

    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        self.tabGeneral.setWidget()  # set tab General (info)
        self.tabLinearRegression.setWidget()  # set tab LinearRegression
        self.tabRidge.setWidget()  # set tab Ridge
        self.tabLasso.setWidget()  # set tab Lasso
        self.tabDecisionTreeRegressor.setWidget()  # set tab DecisionTreeRegressor
        self.tabRandomForestRegressor.setWidget()  # set tab RandomForestRegressor
        self.tabGradientBoostingRegressor.setWidget()  # set tab GradientBoostingRegressor
        self.tabAdaBoostRegressor.setWidget()  # set tab AdaBoostRegressor
        self.tabKNeighborsRegressor.setWidget()  # set tab KNeighborsRegressor

        self.mainTabWidget.addTab(self.tabGeneral, "General")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabLinearRegression, "Linear Regression")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabRidge, "Ridge")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabLasso, "Lasso")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabDecisionTreeRegressor, "Decision Tree Regressor")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabRandomForestRegressor, "Random Forest Regressor")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabGradientBoostingRegressor,
                                  "Gradient Boosting Regressor")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabAdaBoostRegressor, "Ada Boost Regressor")  # add tab to mainTabWidget
        self.mainTabWidget.addTab(self.tabKNeighborsRegressor, "K-Neighbors Regressor")  # add tab to mainTabWidget

        self.vbox_main_layout.addWidget(self.mainTabWidget)  # add tabWidget to main layout


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
        self.comboBox_MachineLearningMethods.addItem(MLPF_TYPE_SEQUENTIAL)
        self.comboBox_MachineLearningMethods.addItem(MLPF_TYPE_AVERAGE)
        self.comboBox_MachineLearningMethods.addItem(MLPF_TYPE_SEQUENTIAL_AVERAGE)

        # MultifileTrainingProcessing
        self.comboBox_MultifileTrainingProcessing = QComboBox()
        self.comboBox_MultifileTrainingProcessing.setMinimumWidth(200)
        self.comboBox_MultifileTrainingProcessing.addItem(MLPF_MULTIFILE_TRAINING_PROCESSING_LINEAR)
        self.comboBox_MultifileTrainingProcessing.addItem(MLPF_MULTIFILE_TRAINING_PROCESSING_PARALLEL)

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
        label_TestPercentage = QLabel("Test Percentage (0.00-0.75):")
        label_HoldOutPercentage = QLabel("Hold-Out Percentage (0.00-0.75):")
        label_SetOutPath = QLabel("Export Folder:")
        label_MachineLearningMethod = QLabel("Machine Learning Method:")
        label_MachineLearningMethodIndex = QLabel("Method Usage Index:")
        label_MultifileTrainingProcessing = QLabel("Multifile Training Processing:")

        # Set SpinBoxes
        hbox_ExperimentalNumber = QHBoxLayout()
        hbox_ExperimentalNumber.addWidget(label_ExperimentalNumber)
        hbox_ExperimentalNumber.addWidget(self.spinBox_ExperimentNumber)
        hbox_ExperimentalNumber.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TestPercentage = QHBoxLayout()
        hbox_TestPercentage.addWidget(label_TestPercentage)
        hbox_TestPercentage.addWidget(self.doubleSpinBox_TestPercentage)
        hbox_TestPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_HoldOutPercentage = QHBoxLayout()
        hbox_HoldOutPercentage.addWidget(label_HoldOutPercentage)
        hbox_HoldOutPercentage.addWidget(self.doubleSpinBox_HoldoutPercentage)
        hbox_HoldOutPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_finalSpinBoxes = QVBoxLayout()
        vbox_finalSpinBoxes.addLayout(hbox_ExperimentalNumber)
        vbox_finalSpinBoxes.addLayout(hbox_TestPercentage)
        vbox_finalSpinBoxes.addLayout(hbox_HoldOutPercentage)

        # SetOutPath Layout
        hbox_SetOutPath = QHBoxLayout()
        hbox_SetOutPath.addWidget(label_SetOutPath)
        hbox_SetOutPath.addWidget(self.lineEdit_SetOutPath)
        hbox_SetOutPath.addWidget(self.buttonSetOutPath)

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
        self.vbox_main_layout.addSpacerItem(QSpacerItem(0, INT_MAX_STRETCH))
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

    def setEvents_(self):
        self.buttonRestoreDefault.clicked.connect(self.restoreDefaultValues)

    # ------------------------------ #
    # ----- GET DEFAULT VALUES ----- #
    # ------------------------------ #
    def getDefaultExperimentNumber(self):
        return self._experimentNumberDefaultValue

    def getDefaultTestPercentage(self):
        return self._testPercentageDefaultValue

    def getDefaultHoldoutPercentage(self):
        return self._holdoutPercentageDefaultValue

    def getDefaultExportPath(self):
        return self._exportPathDefaultValue

    def getDefaultMethod(self):
        return self._mlMethodDefaultValue

    def getDefaultMethodIndex(self):
        return self._mlMethodIndexDefaultValue

    def getDefaultMultifileTrainingProcessing(self):
        return self._mlMultifileTrainingProcessingDefaultValue


class WidgetTabMachineLearningSettingsLinearRegression(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsRidge(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsLasso(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsDecisionTreeRegressor(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsRandomForestRegressor(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsGradientBoostingRegressor(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsAdaBoostRegressor(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


class WidgetTabMachineLearningSettingsKNeighborsRegressor(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        # Set Label
        pass


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
