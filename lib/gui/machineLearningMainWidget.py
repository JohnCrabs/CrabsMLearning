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
    QLineEdit
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
        self._DKEY_MLP_SEQUENCE_STEP_INDEX: str = 'sequence-step-index'

        # -------------------------------- #
        # ----- Set QTabWidget ----------- #
        # -------------------------------- #

        self.mainTabWidget = QTabWidget()  # Create a Tab Widget
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
            self.dkey_mlpSequenceStepIndex(): 7
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

    def dkey_mlpSequenceStepIndex(self):
        return self._DKEY_MLP_SEQUENCE_STEP_INDEX

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setTab(self):
        pass

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
        pass

    def setTabSettingsGeneralEvents_(self):
        # Spin Boxes Events
        self.widgetTabMachineLearningSettings.tabGeneral.spinBox_ExperimentNumber.valueChanged.connect(
            self.actionExperimentResultChange)

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
        pass

    def actionButtonRemove(self):
        if self.listWidget_FileList.currentItem() is not None:  # if some item is selected
            self.dict_tableFilesPaths.pop(self.fileName, None)  # Delete item from dict
            self.listWidget_FileList.takeItem(self.listWidget_FileList.currentRow())  # Delete item from widget
            self.actionFileListRowChanged_event()  # run the row changed event
            self.updateButtonRemove()

            # if there are not enough files loaded
            if self.dict_tableFilesPaths.keys().__len__() < 1:
                self.buttonExecute.setEnabled(False)  # disable the Execute Button

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

        # ------------------------------ #
        # ----- Set Default Values ----- #
        # ------------------------------ #
        self._experimentNumberDefaultValue = MLF_DEFAULT_EXPERIMENT_VALUE
        self._testPercentageDefaultValue = MLF_DEFAULT_TEST_PERCENTAGE
        self._holdoutPercentageDefaultValue = MLF_DEFAULT_HOLDOUT_PERCENTAGE
        self._exportPathDefaultValue = MLF_DEFAULT_EXPORT_FOLDER_PATH

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

        self.vbox_main_layout.addLayout(vbox_finalSpinBoxes)
        self.vbox_main_layout.addLayout(hbox_SetOutPath)
        self.vbox_main_layout.addSpacerItem(QSpacerItem(0, INT_MAX_STRETCH))
        self.vbox_main_layout.addLayout(hbox_buttons)

    def restoreDefaultValues(self):
        # set default value
        self.spinBox_ExperimentNumber.setValue(self._experimentNumberDefaultValue)
        self.doubleSpinBox_TestPercentage.setValue(self._testPercentageDefaultValue)
        self.doubleSpinBox_HoldoutPercentage.setValue(self._holdoutPercentageDefaultValue)
        self.lineEdit_SetOutPath.setText(os.path.normpath(self._exportPathDefaultValue))

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
