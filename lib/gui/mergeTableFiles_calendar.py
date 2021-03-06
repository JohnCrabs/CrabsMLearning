import sys
import os
import tkinter as tk
from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget, QListWidgetItem, QFileDialog, QLabel, QTabWidget, QComboBox, QCheckBox, QRadioButton, QLineEdit, \
    QButtonGroup, QSpinBox
from PySide2.QtGui import QIcon, QPixmap, QFont

import lib.core.file_manipulation as file_manip
import lib.core.my_calendar_v2 as my_cal_v2


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
_DKEY_COLUMN_DELIMITER = 'column-delimiter'
_DKEY_DATE_COLUMN = 'date-column'
_DKEY_TIME_COLUMN = 'time-column'
_DKEY_PRIMARY_COLUMN = 'primary-column'
_DKEY_EVENT_COLUMNS = 'event-columns'
_DKEY_CHANGE_FILE_DATE_FORMAT_STATE = 'change-file-date-format'
_DKEY_FILE_DATE_FORMAT = 'file-date-format'
_DKEY_NEW_FILE_DATE_FORMAT = 'new-file-date-format'
_DKEY_FILE_DATE_DELIMITER = 'file-date-delimiter'
_DKEY_NEW_FILE_DATE_DELIMITER = 'new-file-date-delimiter'


_DKEY_MYCALV2_START_YEAR = 'start-year'
_DKEY_MYCALV2_END_YEAR = 'end-year'
_DKEY_MYCALV2_TIME_COLUMN_STATE = 'time-column-state'
_DKEY_MYCALV2_DATE_FORMAT = 'date-format'
_DKEY_MYCALV2_DATE_DELIMITER = 'date-delimiter'

_MYCALV2_DEFAULT_YEAR = my_cal_v2.getCurrentYear()
_MYCALV2_DEFAULT_DATE_FORMAT = my_cal_v2.DD_MM_YYYY
_MYCALV2_DEFAULT_DATE_DELIM = my_cal_v2.del_dash


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


class WidgetMergeTableFilesCalendar(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()

        self.setStyleSheet(setStyle_())  # Set the styleSheet

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

        # -------------------------------- #
        # ----- Set QTabWidget ----------- #
        # -------------------------------- #

        self.mainTabWidget = QTabWidget()  # Create a Tab Widget
        self.widgetTabFileManagement = WidgetTabFileManagement()  # Create the first Tab of TabWidget
        self.widgetTabDate = WidgetTabDate()  # Create a calendar Tab for TabWidget
        self.widgetTabMyCalendarOptions = WidgetMyCalendarOptions()  # Create a calendar Option Tab

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

        self.buttonGenerate = QPushButton("Generate")
        self.buttonGenerate.setMinimumWidth(0)  # Set Minimum Width
        self.buttonGenerate.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonGenerate.setToolTip('Open the generate window to proceed in the merging process.')  # Add Description

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
        self.dict_myCalendar_v2_settings = {}  # a dictionary to store the calendar settings
        self.setCalendarV2settings()

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        """
        A function to create the widget components into the main QWidget
        :return: Nothing
        """
        # Disable Generate Button
        self.buttonGenerate.setEnabled(False)

        # Set Column vbox
        labelColumnList = QLabel("Column List:")
        vbox_listColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listColumns.addWidget(labelColumnList)  # Add Label
        vbox_listColumns.addWidget(self.listWidget_ColumnList)  # Add Column List

        # Set add/remove button in vbox
        hbox_listFileButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listFileButtons.addWidget(self.buttonAdd)  # Add buttonAdd
        hbox_listFileButtons.addWidget(self.buttonRemove)  # Add buttonRemove
        hbox_listFileButtons.addWidget(self.buttonGenerate)  # Add buttonGenerate

        # Set FileList in hbox
        labelFileList = QLabel("Opened File List:")
        vbox_listFile = QVBoxLayout()  # Create a Vertical Box Layout
        vbox_listFile.addWidget(labelFileList)  # Add Label
        vbox_listFile.addWidget(self.listWidget_FileList)  # Add FileList
        vbox_listFile.addLayout(vbox_listColumns)  # Add listColumns
        vbox_listFile.addLayout(hbox_listFileButtons)  # Add vbox_listFileButtons layout

        # Set main Tab Widget
        self.widgetTabFileManagement.setWidget()  # Set the Tab File Management Widget
        self.mainTabWidget.addTab(self.widgetTabFileManagement, "File Management")  # Add it to mainTanWidget
        self.widgetTabDate.setWidget()  # Set the Tab Date Widget
        self.mainTabWidget.addTab(self.widgetTabDate, "Date Column Management")  # Add it to mainTabWidget
        self.widgetTabMyCalendarOptions.setWidget()  # Set my Calendar Option Tab
        self.mainTabWidget.addTab(self.widgetTabMyCalendarOptions, "MyCalendar_V2 Settings")

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

    # ---------------------------------- #
    # ----- Reuse Action Functions ----- #
    # ---------------------------------- #

    def addItemsToList(self, fullPath, splitter=my_cal_v2.del_comma):
        fileName = fullPath.split('/')[-1:][0]  # find the name of the file
        # Create the dictionary
        self.dict_tableFilesPaths[fileName] = {_DKEY_FILE_NAME: fileName,
                                               _DKEY_FULLPATH: fullPath,
                                               _DKEY_COLUMNS: file_manip.getColumnNames(fullPath, splitter=splitter),
                                               _DKEY_COLUMN_DELIMITER: splitter,
                                               _DKEY_DATE_COLUMN: None,
                                               _DKEY_TIME_COLUMN: None,
                                               _DKEY_PRIMARY_COLUMN: None,
                                               _DKEY_EVENT_COLUMNS: [],
                                               _DKEY_FILE_DATE_FORMAT: None,
                                               _DKEY_FILE_DATE_DELIMITER: '',
                                               _DKEY_CHANGE_FILE_DATE_FORMAT_STATE: False,
                                               _DKEY_NEW_FILE_DATE_FORMAT: None,
                                               _DKEY_NEW_FILE_DATE_DELIMITER: ''
                                               }
        self.listWidget_FileList.addItem(QListWidgetItem(fileName))  # Add Item to List

    def resetDateColumn(self, fileName):
        # Set DATE_COLUMN for the specified FILE to None
        self.dict_tableFilesPaths[fileName][_DKEY_DATE_COLUMN] = None

    def resetTimeColumn(self, fileName):
        # Set DATE_COLUMN for the specified FILE to None
        self.dict_tableFilesPaths[fileName][_DKEY_TIME_COLUMN] = None

    def resetPrimEventColumn(self, fileName):
        # Set PRIMARY_COLUMN for the specified FILE to None
        self.dict_tableFilesPaths[fileName][_DKEY_PRIMARY_COLUMN] = None

    def resetEventColumn(self, fileName):
        # Set EVENT_COLUMN for the specified FILE to [] -> EMPTY_LIST
        self.dict_tableFilesPaths[fileName][_DKEY_EVENT_COLUMNS] = []

    def removeFromEventColumn(self, fileName, column):
        # Remove the specified COLUMN from EVENT_COLUMN_LIST for the specified FILE
        if column in self.dict_tableFilesPaths[fileName][_DKEY_EVENT_COLUMNS]:
            self.dict_tableFilesPaths[fileName][_DKEY_EVENT_COLUMNS].remove(column)

    def updateDateList(self):
        self.widgetTabFileManagement.listWidget_DateColumns.clear()  # Clear Date Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key  (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_DATE_COLUMN] is not None:  # if date key is not None
                # Add ITEM to DATE widget
                self.widgetTabFileManagement.listWidget_DateColumns.addItem(
                    QListWidgetItem(key + " -> " + self.dict_tableFilesPaths[key][_DKEY_DATE_COLUMN]))

    def updateTimeList(self):
        self.widgetTabFileManagement.listWidget_TimeColumns.clear()  # Clear Date Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key  (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_TIME_COLUMN] is not None:  # if date key is not None
                # Add ITEM to DATE widget
                self.widgetTabFileManagement.listWidget_TimeColumns.addItem(
                    QListWidgetItem(key + " -> " + self.dict_tableFilesPaths[key][_DKEY_TIME_COLUMN]))

    def updatePrimaryEventList(self):
        self.widgetTabFileManagement.listWidget_PrimEventColumns.clear()  # Clear Primary Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_PRIMARY_COLUMN] is not None:  # if primary event key is not None
                # Add ITEM to PRIMARY_EVENT widget
                self.widgetTabFileManagement.listWidget_PrimEventColumns.addItem(
                    QListWidgetItem(key + " -> " + self.dict_tableFilesPaths[key][_DKEY_PRIMARY_COLUMN]))

    def updateEventsList(self):
        self.widgetTabFileManagement.listWidget_EventColumns.clear()  # Clear Event Widget
        for key in self.dict_tableFilesPaths.keys():  # For each key (fileName)
            if self.dict_tableFilesPaths[key][_DKEY_EVENT_COLUMNS] is not []:  # if event key is not []
                for event in self.dict_tableFilesPaths[key][_DKEY_EVENT_COLUMNS]:  # for each EVENT
                    # Add ITEM to EVENT widget
                    self.widgetTabFileManagement.listWidget_EventColumns.addItem(
                        QListWidgetItem(key + " -> " + event))

    def updateWidgetTabDate(self):
        if self.fileName is not None:
            if self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_FORMAT] is None:
                self.widgetTabDate.comboBox_DateFileFormat.setCurrentIndex(0)
            else:
                item = self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_FORMAT]
                self.widgetTabDate.comboBox_DateFileFormat.setCurrentText(item)

            if self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT] is None:
                self.widgetTabDate.comboBox_ChangeDateFileFormat.setCurrentIndex(0)
            else:
                item = self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT]
                self.widgetTabDate.comboBox_ChangeDateFileFormat.setCurrentText(item)

            state = self.dict_tableFilesPaths[self.fileName][_DKEY_CHANGE_FILE_DATE_FORMAT_STATE]
            self.widgetTabDate.checkBox_ChangeDateFileFormat.setChecked(state)
            self.widgetTabDate.comboBox_ChangeDateFileFormat.setEnabled(state)

            if self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER] == my_cal_v2.del_slash:
                self.widgetTabDate.lineEdit_DateFileCustom.setText('')
                self.widgetTabDate.radioButton_DateFileSlash.setChecked(True)
            elif self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER] == my_cal_v2.del_dash:
                self.widgetTabDate.lineEdit_DateFileCustom.setText('')
                self.widgetTabDate.radioButton_DateFileDash.setChecked(True)
            else:
                tmp_text = self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER]
                self.widgetTabDate.lineEdit_DateFileCustom.setText(tmp_text)
                self.widgetTabDate.radioButton_DateFileCustom.setChecked(True)

            if self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER] == my_cal_v2.del_slash:
                self.widgetTabDate.lineEdit_NewDateFileCustom.setText('')
                self.widgetTabDate.radioButton_NewDateFileSlash.setChecked(True)
            elif self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER] == my_cal_v2.del_dash:
                self.widgetTabDate.lineEdit_NewDateFileCustom.setText('')
                self.widgetTabDate.radioButton_NewDateFileDash.setChecked(True)
            else:
                tmp_text = self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER]
                self.widgetTabDate.lineEdit_NewDateFileCustom.setText(tmp_text)
                self.widgetTabDate.radioButton_NewDateFileCustom.setChecked(True)

    def setCalendarV2settings(self):
        self.dict_myCalendar_v2_settings = {_DKEY_MYCALV2_START_YEAR: _MYCALV2_DEFAULT_YEAR.__int__(),
                                            _DKEY_MYCALV2_END_YEAR: _MYCALV2_DEFAULT_YEAR.__int__(),
                                            _DKEY_MYCALV2_DATE_FORMAT: _MYCALV2_DEFAULT_DATE_FORMAT,
                                            _DKEY_MYCALV2_DATE_DELIMITER: _MYCALV2_DEFAULT_DATE_DELIM}

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
        self.buttonGenerate.clicked.connect(self.actionButtonGenerate)  # buttonGenerate -> clicked

        # buttonDateColumn -> clicked
        self.widgetTabFileManagement.buttonDateColumn.clicked.connect(self.actionButtonDateColumn)
        # buttonRemDateColumn -> clicked
        self.widgetTabFileManagement.buttonRemDateColumn.clicked.connect(self.actionButtonRemDateColumn)
        # buttonTimeColumn -> clicked
        self.widgetTabFileManagement.buttonTimeColumn.clicked.connect(self.actionButtonTimeColumn)
        # buttonRemTimeColumn -> clicked
        self.widgetTabFileManagement.buttonRemTimeColumn.clicked.connect(self.actionButtonRemTimeColumn)
        # buttonPrimaryEvent -> clicked
        self.widgetTabFileManagement.buttonPrimaryEvent.clicked.connect(self.actionButtonPrimaryEvent)
        # buttonRemPrimaryEvent -> clicked
        self.widgetTabFileManagement.buttonRemPrimaryEvent.clicked.connect(self.actionButtonRemPrimaryEvent)
        # buttonEvent -> clicked
        self.widgetTabFileManagement.buttonEvent.clicked.connect(self.actionButtonEvent)
        # buttonRemEvent -> clicked
        self.widgetTabFileManagement.buttonRemEvent.clicked.connect(self.actionButtonRemEvent)

        # ListWidget Events
        self.listWidget_FileList.currentRowChanged.connect(self.actionFileListRowChanged_event)

        # checkBox_ChangeDateFileFormat Event
        self.widgetTabDate.checkBox_ChangeDateFileFormat.stateChanged.connect(self.actionChangeFileDateFormat)

        # comboBox_DateFileFormat Event
        self.widgetTabDate.comboBox_DateFileFormat.currentTextChanged.connect(self.actionComboBoxFileDateFormatChanged)
        # comboBox_ChangeDateFileFormat Event
        self.widgetTabDate.comboBox_ChangeDateFileFormat.currentTextChanged.connect(
            self.actionComboBoxChangeFileDateFormatChanged)

        # radioButton_DateFileSlash
        self.widgetTabDate.radioButton_DateFileSlash.toggled.connect(self.actionRadioButtFileDelSlash)
        # radioButton_DateFileDash
        self.widgetTabDate.radioButton_DateFileDash.toggled.connect(self.actionRadioButtFileDelDash)
        # radioButton_DateFileCustom
        self.widgetTabDate.radioButton_DateFileCustom.toggled.connect(self.actionRadioButtCustom)
        # lineEdit_DateFileCustom
        self.widgetTabDate.lineEdit_DateFileCustom.textChanged.connect(self.actionRadioButtCustom)

        # radioButton_DateFileSlash
        self.widgetTabDate.radioButton_NewDateFileSlash.toggled.connect(self.actionRadioButtFileDelNewSlash)
        # radioButton_DateFileDash
        self.widgetTabDate.radioButton_NewDateFileDash.toggled.connect(self.actionRadioButtFileDelNewDash)
        # radioButton_DateFileCustom
        self.widgetTabDate.radioButton_NewDateFileCustom.toggled.connect(self.actionRadioButtNewCustom)
        # lineEdit_DateFileCustom
        self.widgetTabDate.lineEdit_NewDateFileCustom.textChanged.connect(self.actionRadioButtNewCustom)

        # spinBox_startYear
        self.widgetTabMyCalendarOptions.spinBox_startYear.textChanged.connect(self.actionSpinBoxStartYear)
        # spinBox_endYear
        self.widgetTabMyCalendarOptions.spinBox_endYear.textChanged.connect(self.actionSpinBoxEndYear)
        # comboBox_DateFormat
        self.widgetTabMyCalendarOptions.comboBox_DateFormat.currentTextChanged.connect(self.actionComboBoxMyCalV2Date)
        # radioButton_DateSlash
        self.widgetTabMyCalendarOptions.radioButton_DateSlash.toggled.connect(self.actionRadioButtMyCalV2DateDelimSlash)
        # radioButton_DateDash
        self.widgetTabMyCalendarOptions.radioButton_DateDash.toggled.connect(self.actionRadioButtMyCalV2DateDelimDash)
        # radioButton_DateCustom
        self.widgetTabMyCalendarOptions.radioButton_DateCustom.toggled.connect(
            self.actionRadioButtMyCalV2DateDelimCustom)
        # lineEdit_DateCustom
        self.widgetTabMyCalendarOptions.lineEdit_DateCustom.textChanged.connect(
            self.actionRadioButtMyCalV2DateDelimCustom)

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

            if self.dict_tableFilesPaths.keys().__len__() >= 2:
                self.buttonGenerate.setEnabled(True)
            # self.prt_dict_tableFilePaths()

    def actionButtonRemove(self):
        if self.listWidget_FileList.currentItem() is not None:  # if some item is selected
            self.dict_tableFilesPaths.pop(self.fileName, None)  # Delete item from dict
            self.listWidget_FileList.takeItem(self.listWidget_FileList.currentRow())  # Delete item from widget
            self.actionFileListRowChanged_event()  # run the row changed event
            self.updateDateList()  # update DATE widget
            self.updatePrimaryEventList()  # update PRIMARY_EVENT widget
            self.updateEventsList()  # update EVENT widget

            if self.dict_tableFilesPaths.keys().__len__() < 2:
                self.buttonGenerate.setEnabled(False)

    def actionButtonGenerate(self):
        list_primary_event = []  # create a list to store the primary events
        list_of_events = []  # create list to store the actual events
        list_of_year = []  # a list to store the years for the calendar

        myCalV2_DateFormat = self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_FORMAT]
        myCalV2_DateDelimiter = self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_DELIMITER]

        def set_list_primary_event(listInput, listOfSelectedEvents, tmpPrimaryEventIndex=0):
            """
            A tmp function to create the list_primary_event and the list_of_events
            :param listInput: The list of all data
            :param listOfSelectedEvents: The list of selected event columns
            :param tmpPrimaryEventIndex:  The primary event index
            :return: Nothing
            """
            for event_name in listInput[0]:  # for each event_name (column) in data list
                if event_name in listOfSelectedEvents and event_name not in list_of_events:
                    list_of_events.append(event_name)
            for index in range(1, len(listInput)):
                if listInput[index][tmpPrimaryEventIndex] not in list_primary_event:
                    list_primary_event.append(listInput[index][tmpPrimaryEventIndex])

        dict_primary_event_index = {}  # a dictionary to store the primary event indexes
        dict_date_index = {}   # a dictionary to store the date column indexes
        if self.dict_tableFilesPaths.keys().__len__() >= 2:  # if there are at least 2 files (safety if)
            # Error Checking
            bool_add_primary_event = True
            bool_add_date_column = True
            for fileName in self.dict_tableFilesPaths.keys():
                if self.dict_tableFilesPaths[fileName][_DKEY_PRIMARY_COLUMN] is not None:
                    primary_event = self.dict_tableFilesPaths[fileName][_DKEY_PRIMARY_COLUMN]
                    primary_event_index = self.dict_tableFilesPaths[fileName][_DKEY_COLUMNS].index(primary_event)
                    dict_primary_event_index[fileName] = primary_event_index
                    if bool_add_primary_event:
                        list_of_events.append(primary_event)
                        bool_add_primary_event = False
                else:
                    print("ERROR: <", fileName, "> has no Primary event specified!")
                    return

                if self.dict_tableFilesPaths[fileName][_DKEY_DATE_COLUMN] is not None:
                    date_column = self.dict_tableFilesPaths[fileName][_DKEY_DATE_COLUMN]
                    date_index = self.dict_tableFilesPaths[fileName][_DKEY_COLUMNS].index(date_column)
                    dict_date_index[fileName] = date_index
                    if bool_add_date_column:
                        list_of_events.append(date_column)
                        bool_add_date_column = False
                else:
                    print("ERROR: <", fileName, "> has not a Date column specified!")
                    return

                if self.dict_tableFilesPaths[fileName][_DKEY_EVENT_COLUMNS].__len__() == 0:
                    print("ERROR: <", fileName, "> has not a single Event specified!")
                    return

            # Set Primary Event List and Event List
            for fileName in self.dict_tableFilesPaths.keys():
                filePath = self.dict_tableFilesPaths[fileName][_DKEY_FULLPATH]
                delimiter = self.dict_tableFilesPaths[fileName][_DKEY_COLUMN_DELIMITER]
                fileData = my_cal_v2.read_csv(filePath, delimiter)
                list_of_selected_events = self.dict_tableFilesPaths[fileName][_DKEY_EVENT_COLUMNS]
                set_list_primary_event(fileData, list_of_selected_events, dict_primary_event_index[fileName])

            # Create the Calendar
            # We need to write code (widget to set up this parameters) **************************************
            for i in range(self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_START_YEAR],
                           self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_END_YEAR] + 1):
                list_of_year.append(i)
            # print(list_of_year)

            event_calendar = my_cal_v2.MyCalendar(list_of_years=list_of_year, is_time=False,
                                                  date_format=myCalV2_DateFormat,
                                                  date_delimiter=myCalV2_DateDelimiter,
                                                  time_format=my_cal_v2.HH_MM,
                                                  time_delimiter=my_cal_v2.del_colon,
                                                  hour_start=0, hour_end=0, hour_step=1,
                                                  minute_start=0, minute_end=0, minute_step=0)
            event_calendar.add_list_key_event_to_calendar(list_key_event=list_primary_event)
            event_calendar.add_list_key_event_to_calendar(list_key_event=list_primary_event,
                                                          list_of_headers=list_of_events)

            # ***********************************************************************************************

            # Add events to Calender (we may need to edit calendar lib so to be fully compatible with the interface)
            for fileName in self.dict_tableFilesPaths.keys():
                print(fileName)
                filePath = self.dict_tableFilesPaths[fileName][_DKEY_FULLPATH]
                delimiter = self.dict_tableFilesPaths[fileName][_DKEY_COLUMN_DELIMITER]
                fileData = my_cal_v2.read_csv(filePath, delimiter)
                event_calendar.add_events_to_calendar(list_of_events=fileData,
                                                      date_index=dict_date_index[fileName],
                                                      time_index=None,
                                                      first_row_header=True,
                                                      list_of_headers=None,
                                                      event_index=dict_primary_event_index[fileName],
                                                      add_events_in_this_list=list_of_events)

            # Export Final List
            # We need to write code (widget to set up this parameters) **************************************
            list_calendar = event_calendar.dict_to_list(date_range=['2020-01-01', '2021-09-20'])
            my_cal_v2.write_csv(csv_path=_PROJECT_FOLDER + "/export_folder/netherlands_RNA.csv",
                                list_write=list_calendar, delimiter=my_cal_v2.del_comma)
            # ***********************************************************************************************
            # event_calendar.print(5)

            print("Finished Successfully!")

        else:
            print("ERROR: The needed number of file is at least 2!")
            return

    def actionButtonDateColumn(self):
        # If some file is selected and some column is selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current column name
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            # If this column exist in the time list
            if self.dict_tableFilesPaths[self.fileName][_DKEY_TIME_COLUMN] == currentColumnSelected:
                self.resetTimeColumn(self.fileName)  # Reset primary event list
                self.updateTimeList()  # update Primary Event widget
            # If this column exist in the primary event list
            elif self.dict_tableFilesPaths[self.fileName][_DKEY_PRIMARY_COLUMN] == currentColumnSelected:
                self.resetPrimEventColumn(self.fileName)  # Reset primary event list
                self.updatePrimaryEventList()  # update Primary Event widget
            # else if this column exist in the event list
            elif currentColumnSelected in self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS]:
                self.removeFromEventColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateEventsList()  # update Event widget

            # print(currentFileName, " -> ", currentColumnSelected)

            # Add it to the DATE_COLUMN
            self.dict_tableFilesPaths[self.fileName][_DKEY_DATE_COLUMN] = currentColumnSelected
            self.updateDateList()  # update Date widget

    def actionButtonRemDateColumn(self):
        # If some file is selected and some columns are selected
        if self.widgetTabFileManagement.listWidget_DateColumns.isActiveWindow() and \
                self.widgetTabFileManagement.listWidget_DateColumns.currentItem() is not None:
            # get selected items
            selectedItems = self.widgetTabFileManagement.listWidget_DateColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                self.resetDateColumn(fileName)  # remove DATE from the list
            self.updateDateList()  # update DATE widget

    def actionButtonTimeColumn(self):
        # If some file is selected and some column is selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current column name
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            # If this column exist in the date list
            if self.dict_tableFilesPaths[self.fileName][_DKEY_DATE_COLUMN] == currentColumnSelected:
                self.resetDateColumn(self.fileName)  # Reset date list
                self.updateDateList()  # update Date widget
            # If this column exist in the primary event list
            elif self.dict_tableFilesPaths[self.fileName][_DKEY_PRIMARY_COLUMN] == currentColumnSelected:
                self.resetPrimEventColumn(self.fileName)  # Reset primary event list
                self.updatePrimaryEventList()  # update Primary Event widget
            # else if this column exist in the event list
            elif currentColumnSelected in self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS]:
                self.removeFromEventColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateEventsList()  # update Event widget

            # print(currentFileName, " -> ", currentColumnSelected)

            # Add it to the DATE_COLUMN
            self.dict_tableFilesPaths[self.fileName][_DKEY_TIME_COLUMN] = currentColumnSelected
            self.updateTimeList()  # update Date widget

    def actionButtonRemTimeColumn(self):
        # If some file is selected and some columns are selected
        if self.widgetTabFileManagement.listWidget_DateColumns.isActiveWindow() and \
                self.widgetTabFileManagement.listWidget_TimeColumns.currentItem() is not None:
            # get selected items
            selectedItems = self.widgetTabFileManagement.listWidget_TimeColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                self.resetTimeColumn(fileName)  # remove DATE from the list
            self.updateTimeList()  # update DATE widget

    def actionButtonPrimaryEvent(self):
        # If some file is selected and some column is selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current column name
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            # If this column exist in the date list
            if self.dict_tableFilesPaths[self.fileName][_DKEY_DATE_COLUMN] == currentColumnSelected:
                self.resetDateColumn(self.fileName)  # Reset date list
                self.updateDateList()  # update Date widget
            # If this column exist in the time list
            elif self.dict_tableFilesPaths[self.fileName][_DKEY_TIME_COLUMN] == currentColumnSelected:
                self.resetTimeColumn(self.fileName)  # Reset primary event list
                self.updateTimeList()  # update Primary Event widget
            # else if this column exist in the event list
            elif currentColumnSelected in self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS]:
                self.removeFromEventColumn(self.fileName, currentColumnSelected)  # remove it from the list
                self.updateEventsList()  # update Event widget

            # print(currentFileName, " -> ", currentColumnSelected)

            # Add it to the PRIMARY_EVENT
            self.dict_tableFilesPaths[self.fileName][_DKEY_PRIMARY_COLUMN] = currentColumnSelected
            self.updatePrimaryEventList()  # update Primary Event widget

    def actionButtonRemPrimaryEvent(self):
        # If some file is selected and some columns are selected
        if self.widgetTabFileManagement.listWidget_PrimEventColumns.isActiveWindow() and \
                self.widgetTabFileManagement.listWidget_PrimEventColumns.currentItem() is not None:
            # get selected item
            selectedItems = self.widgetTabFileManagement.listWidget_PrimEventColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                self.resetPrimEventColumn(fileName)  # remove PRIMARY_EVENT from the list
            self.updatePrimaryEventList()  # update PRIMARY_EVENT widget

    def actionButtonEvent(self):
        # If some file is selected and some columns are selected
        if self.listWidget_FileList.currentItem() is not None and \
                self.listWidget_ColumnList.currentItem() is not None:
            # get current columns selected
            currentSelectedItems = self.listWidget_ColumnList.selectedItems()
            for currentColumnSelected in currentSelectedItems:  # for each item selected
                # If this column exist in the date list
                if self.dict_tableFilesPaths[self.fileName][_DKEY_DATE_COLUMN] == currentColumnSelected.text():
                    self.resetDateColumn(self.fileName)  # Reset date list
                    self.updateDateList()  # update Date widget
                # If this column exist in the time list
                elif self.dict_tableFilesPaths[self.fileName][_DKEY_TIME_COLUMN] == currentColumnSelected:
                    self.resetTimeColumn(self.fileName)  # Reset primary event list
                    self.updateTimeList()  # update Primary Event widget
                # else if this column exist in the primary event list
                elif self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS] == currentColumnSelected.text():
                    self.resetPrimEventColumn(self.fileName)  # Reset primary event list
                    self.updatePrimaryEventList()  # update Primary Event widget

                # if this column is not in the EVENT List
                if currentColumnSelected.text() not in self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS]:
                    # Add it to list
                    self.dict_tableFilesPaths[self.fileName][_DKEY_EVENT_COLUMNS].append(currentColumnSelected.text())

                # print(currentFileName, " -> ", currentColumnSelected.text())

            self.updateEventsList()  # update Event widget

    def actionButtonRemEvent(self):
        # If some file is selected and some columns are selected
        if self.widgetTabFileManagement.listWidget_EventColumns.currentItem() is not None:
            # get selected items
            selectedItems = self.widgetTabFileManagement.listWidget_EventColumns.selectedItems()
            for item in selectedItems:  # for each item
                tmp_str = item.text()  # get text
                fileName = tmp_str.split(' -> ')[0]  # get fileName
                columnName = tmp_str.split(' -> ')[1]  # get columnName
                self.removeFromEventColumn(fileName, columnName)  # remove event from the list
            self.updateEventsList()  # update EVENT widget

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
        self.updateWidgetTabDate()

    def actionChangeFileDateFormat(self):
        state = self.widgetTabDate.checkBox_ChangeDateFileFormat.isChecked()
        self.widgetTabDate.updateNewDateToggleState()
        if self.listWidget_FileList.currentItem() is not None:
            self.dict_tableFilesPaths[self.fileName][_DKEY_CHANGE_FILE_DATE_FORMAT_STATE] = state
            if state:
                if self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT] is None:
                    self.widgetTabDate.comboBox_ChangeDateFileFormat.setCurrentIndex(0)
                else:
                    item = self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT]
                    self.widgetTabDate.comboBox_ChangeDateFileFormat.setCurrentText(item)
            # self.prt_dict_tableFilePaths()

    def actionComboBoxFileDateFormatChanged(self):
        if self.fileName is not None:
            item = self.widgetTabDate.comboBox_DateFileFormat.currentText()
            if item == 'NULL':
                self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_FORMAT] = None
            else:
                self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_FORMAT] = item

    def actionComboBoxChangeFileDateFormatChanged(self):
        if self.fileName is not None:
            item = self.widgetTabDate.comboBox_ChangeDateFileFormat.currentText()
            if item == 'NULL' and self.fileName is not None:
                self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT] = None
            else:
                self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_FORMAT] = item

    def actionRadioButtFileDelSlash(self):
        if self.widgetTabDate.radioButton_DateFileSlash.isChecked() and self.fileName is not None:
            self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER] = my_cal_v2.del_slash

    def actionRadioButtFileDelDash(self):
        if self.widgetTabDate.radioButton_DateFileDash.isChecked() and self.fileName is not None:
            self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER] = my_cal_v2.del_dash

    def actionRadioButtCustom(self):
        if self.widgetTabDate.radioButton_DateFileCustom.isChecked() and self.fileName is not None:
            delim = self.widgetTabDate.lineEdit_DateFileCustom.text()
            self.dict_tableFilesPaths[self.fileName][_DKEY_FILE_DATE_DELIMITER] = delim

    def actionRadioButtFileDelNewSlash(self):
        if self.widgetTabDate.radioButton_NewDateFileSlash.isChecked() and self.fileName is not None:
            self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER] = my_cal_v2.del_slash

    def actionRadioButtFileDelNewDash(self):
        if self.widgetTabDate.radioButton_NewDateFileDash.isChecked() and self.fileName is not None:
            self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER] = my_cal_v2.del_dash

    def actionRadioButtNewCustom(self):
        if self.widgetTabDate.radioButton_NewDateFileCustom.isChecked() and self.fileName is not None:
            delim = self.widgetTabDate.lineEdit_NewDateFileCustom.text()
            self.dict_tableFilesPaths[self.fileName][_DKEY_NEW_FILE_DATE_DELIMITER] = delim

    def actionSpinBoxStartYear(self):
        year = self.widgetTabMyCalendarOptions.spinBox_startYear.text()
        self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_START_YEAR] = int(year.__str__())
        # print(type(self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_START_YEAR]),
        #       self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_START_YEAR])
        # print(type(year), year)

    def actionSpinBoxEndYear(self):
        year = self.widgetTabMyCalendarOptions.spinBox_endYear.text()
        self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_END_YEAR] = int(year.__str__())
        # print(type(self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_END_YEAR]),
        #       self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_END_YEAR])
        # print(type(year), year)

    def actionComboBoxMyCalV2Date(self):
        value = self.widgetTabMyCalendarOptions.comboBox_DateFormat.currentText()
        self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_FORMAT] = value

    def actionRadioButtMyCalV2DateDelimSlash(self):
        if self.widgetTabMyCalendarOptions.radioButton_DateSlash.isChecked():
            self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_DELIMITER] = my_cal_v2.del_slash

    def actionRadioButtMyCalV2DateDelimDash(self):
        if self.widgetTabMyCalendarOptions.radioButton_DateDash.isChecked():
            self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_DELIMITER] = my_cal_v2.del_dash

    def actionRadioButtMyCalV2DateDelimCustom(self):
        if self.widgetTabMyCalendarOptions.radioButton_DateCustom.isChecked():
            delim = self.widgetTabMyCalendarOptions.lineEdit_DateCustom.text()
            self.dict_myCalendar_v2_settings[_DKEY_MYCALV2_DATE_DELIMITER] = delim


class WidgetTabFileManagement(QWidget):
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
        self.buttonDateColumn = QPushButton("Date")
        self.buttonDateColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonDateColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonDateColumn.setShortcut("D")  # Set Shortcut
        self.buttonDateColumn.setToolTip('Set selected column as Date Column.')  # Add Description

        self.buttonRemDateColumn = QPushButton("Remove")
        self.buttonRemDateColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemDateColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemDateColumn.setToolTip('Remove selected column from Date List.')  # Add Description

        self.buttonTimeColumn = QPushButton("Time")
        self.buttonTimeColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonTimeColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonTimeColumn.setShortcut("T")  # Set Shortcut
        self.buttonTimeColumn.setToolTip('Set selected column as Time Column.')  # Add Description

        self.buttonRemTimeColumn = QPushButton("Remove")
        self.buttonRemTimeColumn.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemTimeColumn.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemTimeColumn.setToolTip('Remove selected column from Time List.')  # Add Description

        self.buttonPrimaryEvent = QPushButton("Primary Event")
        self.buttonPrimaryEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonPrimaryEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonPrimaryEvent.setShortcut("P")  # Set Shortcut
        self.buttonPrimaryEvent.setToolTip('Set selected column as Primary Event Column.')  # Add Description

        self.buttonRemPrimaryEvent = QPushButton("Remove")
        self.buttonRemPrimaryEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemPrimaryEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemPrimaryEvent.setToolTip('Remove selected column from Primary Event List.')  # Add Description

        self.buttonEvent = QPushButton("Event Column")
        self.buttonEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonEvent.setShortcut("E")  # Set Shortcut
        self.buttonEvent.setToolTip('Set selected column as Event Column.')  # Add Description

        self.buttonRemEvent = QPushButton("Remove")
        self.buttonRemEvent.setMinimumWidth(_INT_BUTTON_MIN_WIDTH)  # Set Minimum Width
        self.buttonRemEvent.setMinimumHeight(_INT_BUTTON_MIN_WIDTH / 2)  # Set Minimum Height
        self.buttonRemEvent.setToolTip('Remove selected column from Event List.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_DateColumns = QListWidget()
        self.listWidget_DateColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_TimeColumns = QListWidget()
        self.listWidget_TimeColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_PrimEventColumns = QListWidget()
        self.listWidget_PrimEventColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_EventColumns = QListWidget()
        self.listWidget_EventColumns.setSelectionMode(QListWidget.ExtendedSelection)

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        # Set column/remove buttons in vbox
        hbox_listDateButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listDateButtons.addWidget(self.buttonDateColumn)  # Add buttonDate
        hbox_listDateButtons.addWidget(self.buttonRemDateColumn)  # Add buttonRemove

        hbox_listTimeButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listTimeButtons.addWidget(self.buttonTimeColumn)  # Add buttonTime
        hbox_listTimeButtons.addWidget(self.buttonRemTimeColumn)  # Add buttonRemove

        hbox_listPrimEventButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listPrimEventButtons.addWidget(self.buttonPrimaryEvent)  # Add buttonPrimaryEvent
        hbox_listPrimEventButtons.addWidget(self.buttonRemPrimaryEvent)  # Add buttonRemove

        hbox_listEventsButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listEventsButtons.addWidget(self.buttonEvent)  # Add buttonEvent
        hbox_listEventsButtons.addWidget(self.buttonRemEvent)  # Add buttonRemove

        # Set Time vbox
        timeText = "Time Column (One common column from each file. This column is " \
                   "\noptional and needs to check the time option in Calendar Options Tab):"
        labelTimeList = QLabel(timeText)
        vbox_listTimeColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listTimeColumns.addWidget(labelTimeList)  # Add Label
        vbox_listTimeColumns.addWidget(self.listWidget_TimeColumns)  # Add Column List
        vbox_listTimeColumns.addLayout(hbox_listTimeButtons)  # Add Layout

        # Set Date vbox
        labelDateList = QLabel("Date Column\n(one common column for each file):")
        vbox_listDateColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listDateColumns.addWidget(labelDateList)  # Add Label
        vbox_listDateColumns.addWidget(self.listWidget_DateColumns)  # Add Column List
        vbox_listDateColumns.addLayout(hbox_listDateButtons)  # Add Layout

        # Set PrimEvent vbox
        labelPrimEventList = QLabel("Primary Event\n(one common column for each file):")
        vbox_listPrimEventColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listPrimEventColumns.addWidget(labelPrimEventList)  # Add Label
        vbox_listPrimEventColumns.addWidget(self.listWidget_PrimEventColumns)  # Add Column List
        vbox_listPrimEventColumns.addLayout(hbox_listPrimEventButtons)  # Add Layout

        # Set EventColumns vbox
        labelEventColumnsList = QLabel("Other Events to be merged\n(they will be set under primary event):")
        vbox_listEventColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listEventColumns.addWidget(labelEventColumnsList)  # Add Label
        vbox_listEventColumns.addWidget(self.listWidget_EventColumns)  # Add Column List
        vbox_listEventColumns.addLayout(hbox_listEventsButtons)  # Add Layout

        # Combine Column Boxes vbox
        vbox_Combine_1 = QVBoxLayout()
        vbox_Combine_1.addLayout(vbox_listDateColumns)
        vbox_Combine_1.addLayout(vbox_listPrimEventColumns)

        vbox_Combine_2 = QVBoxLayout()
        vbox_Combine_2.addLayout(vbox_listTimeColumns)
        vbox_Combine_2.addLayout(vbox_listEventColumns)

        # Set ListWidget in hbox
        hbox_listWidget = QHBoxLayout()  # Create Horizontal Layout
        hbox_listWidget.addLayout(vbox_Combine_1)  # Add vbox_Combine_1 layout
        hbox_listWidget.addLayout(vbox_Combine_2)  # Add vbox_Combine_2 layout

        self.vbox_main_layout.addLayout(hbox_listWidget)


class WidgetTabDate(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # ------------------------ #
        # ----- Set CheckBox ----- #
        # ------------------------ #
        self.checkBox_ChangeDateFileFormat = QCheckBox("Change Date Format (to):")

        # --------------------------- #
        # ----- Set RadioButton ----- #
        # --------------------------- #
        self.radioButton_DateFileSlash = QRadioButton("Slash (/)")
        self.radioButton_DateFileDash = QRadioButton("Dash (-)")
        self.radioButton_DateFileCustom = QRadioButton("Custom:")

        self.radioButton_NewDateFileSlash = QRadioButton("Slash (/)")
        self.radioButton_NewDateFileDash = QRadioButton("Dash (-)")
        self.radioButton_NewDateFileCustom = QRadioButton("Custom:")

        # ------------------------ #
        # ----- Set LineEdit ----- #
        # ------------------------ #
        self.lineEdit_DateFileCustom = QLineEdit()
        self.lineEdit_DateFileCustom.setMaxLength(3)
        self.lineEdit_DateFileCustom.setToolTip("Can be < >, <,>, <.>, <->, </>, <#>, <;>, <:>, etc.")

        self.lineEdit_NewDateFileCustom = QLineEdit()
        self.lineEdit_NewDateFileCustom.setMaxLength(3)
        self.lineEdit_DateFileCustom.setToolTip("Can be < >, <,>, <.>, <->, </>, <#>, <;>, <:>, etc.")

        # ------------------------ #
        # ----- Set ComboBox ----- #
        # ------------------------ #
        self.comboBox_DateFileFormat = QComboBox()
        self.comboBox_ChangeDateFileFormat = QComboBox()
        self.setComboBoxes()

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        # Create Labels
        label_DateFileFormat = QLabel("Date File Format:")
        label_Delimiter = QLabel("Delimiter:")
        label_NewDelimiter = QLabel("New Delimiter:")

        # Horizontal Layout for date format
        hbox_DateFormat = QHBoxLayout()
        hbox_DateFormat.addWidget(label_DateFileFormat)
        hbox_DateFormat.addWidget(self.comboBox_DateFileFormat)
        hbox_DateFormat.addWidget(self.checkBox_ChangeDateFileFormat)
        hbox_DateFormat.addWidget(self.comboBox_ChangeDateFileFormat)

        # Horizontal Layout for delimiter
        buttGroup_CurrentDateDelim = QButtonGroup(self)
        buttGroup_CurrentDateDelim.addButton(self.radioButton_DateFileSlash)
        buttGroup_CurrentDateDelim.addButton(self.radioButton_DateFileDash)
        buttGroup_CurrentDateDelim.addButton(self.radioButton_DateFileCustom)

        buttGroup_NewDateDelim = QButtonGroup(self)
        buttGroup_NewDateDelim.addButton(self.radioButton_NewDateFileSlash)
        buttGroup_NewDateDelim.addButton(self.radioButton_NewDateFileDash)
        buttGroup_NewDateDelim.addButton(self.radioButton_NewDateFileCustom)

        hbox_CurrentDateDelim = QHBoxLayout()
        hbox_CurrentDateDelim.addWidget(label_Delimiter)
        hbox_CurrentDateDelim.addWidget(self.radioButton_DateFileSlash)
        hbox_CurrentDateDelim.addWidget(self.radioButton_DateFileDash)
        hbox_CurrentDateDelim.addWidget(self.radioButton_DateFileCustom)
        hbox_CurrentDateDelim.addWidget(self.lineEdit_DateFileCustom)

        hbox_NewDateDelim = QHBoxLayout()
        hbox_NewDateDelim.addWidget(label_NewDelimiter)
        hbox_NewDateDelim.addWidget(self.radioButton_NewDateFileSlash)
        hbox_NewDateDelim.addWidget(self.radioButton_NewDateFileDash)
        hbox_NewDateDelim.addWidget(self.radioButton_NewDateFileCustom)
        hbox_NewDateDelim.addWidget(self.lineEdit_NewDateFileCustom)

        hbox_Delimeter = QHBoxLayout()
        hbox_Delimeter.addLayout(hbox_CurrentDateDelim)
        hbox_Delimeter.addLayout(hbox_NewDateDelim)

        # Main Layout
        self.vbox_main_layout.addLayout(hbox_DateFormat)
        self.vbox_main_layout.addLayout(hbox_Delimeter)
        self.vbox_main_layout.addSpacerItem(QSpacerItem(0, _INT_MAX_STRETCH))

        self.updateNewDateToggleState()

    # ------------------- #
    # ----- Setters ----- #
    # ------------------- #
    def setComboBoxes(self):
        # DateFileFormat
        self.comboBox_DateFileFormat.addItem("<NULL>")

        self.comboBox_DateFileFormat.addItem(my_cal_v2.DD_MM_YYYY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.DD_YYYY_MM)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.D_M_YYYY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.D_YYYY_M)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.DD_MM_YY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.DD_YY_MM)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.D_M_YY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.D_YY_M)

        self.comboBox_DateFileFormat.addItem(my_cal_v2.MM_DD_YYYY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.MM_YYYY_DD)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.M_D_YYYY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.M_YYYY_D)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.MM_DD_YY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.MM_YY_DD)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.M_D_YY)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.M_YY_D)

        self.comboBox_DateFileFormat.addItem(my_cal_v2.YYYY_MM_DD)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YYYY_DD_MM)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YYYY_M_D)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YYYY_D_M)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YY_MM_DD)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YY_DD_MM)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YY_M_D)
        self.comboBox_DateFileFormat.addItem(my_cal_v2.YY_D_M)

        # ChangeDateFileFormat
        self.comboBox_ChangeDateFileFormat.addItem("<NULL>")

        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.DD_MM_YYYY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.DD_YYYY_MM)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.D_M_YYYY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.D_YYYY_M)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.DD_MM_YY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.DD_YY_MM)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.D_M_YY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.D_YY_M)

        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.MM_DD_YYYY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.MM_YYYY_DD)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.M_D_YYYY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.M_YYYY_D)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.MM_DD_YY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.MM_YY_DD)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.M_D_YY)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.M_YY_D)

        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YYYY_MM_DD)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YYYY_DD_MM)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YYYY_M_D)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YYYY_D_M)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YY_MM_DD)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YY_DD_MM)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YY_M_D)
        self.comboBox_ChangeDateFileFormat.addItem(my_cal_v2.YY_D_M)

    def updateNewDateToggleState(self):
        state = self.checkBox_ChangeDateFileFormat.isChecked()
        self.comboBox_ChangeDateFileFormat.setEnabled(state)

        self.radioButton_NewDateFileSlash.setEnabled(state)
        self.radioButton_NewDateFileDash.setEnabled(state)
        self.radioButton_NewDateFileCustom.setEnabled(state)

        self.lineEdit_NewDateFileCustom.setEnabled(state)


class WidgetMyCalendarOptions(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # ------------------------ #
        # ----- Set QSpinBox ----- #
        # ------------------------ #
        self.spinBox_startYear = QSpinBox()
        self.spinBox_startYear.setMaximum(9999)
        self.spinBox_startYear.setMinimum(0)
        self.spinBox_startYear.setValue(_MYCALV2_DEFAULT_YEAR)
        self.spinBox_endYear = QSpinBox()
        self.spinBox_endYear.setMaximum(9999)
        self.spinBox_endYear.setMinimum(0)
        self.spinBox_endYear.setValue(_MYCALV2_DEFAULT_YEAR)

        # --------------------------- #
        # ----- Set RadioButton ----- #
        # --------------------------- #
        self.radioButton_DateSlash = QRadioButton("Slash (/)")
        self.radioButton_DateSlash.setChecked(True)
        self.radioButton_DateDash = QRadioButton("Dash (-)")
        self.radioButton_DateCustom = QRadioButton("Custom:")

        # ------------------------ #
        # ----- Set ComboBox ----- #
        # ------------------------ #
        self.comboBox_DateFormat = QComboBox()
        self.comboBox_DateFormat.setMinimumWidth(100)
        self.setComboBoxes()

        # ------------------------ #
        # ----- Set LineEdit ----- #
        # ------------------------ #
        self.lineEdit_DateCustom = QLineEdit()
        self.lineEdit_DateCustom.setMaxLength(3)
        self.lineEdit_DateCustom.setMaximumWidth(100)
        self.lineEdit_DateCustom.setToolTip("Can be < >, <,>, <.>, <->, </>, <#>, <;>, <:>, etc.")

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        # Set labels
        boldFont = QFont()
        boldFont.setBold(True)

        label_yearSection = QLabel("Set the year range:")
        label_yearSection.setFont(boldFont)
        label_StartYear = QLabel("Start Year:")
        label_EndYear = QLabel("End Year:")

        label_timeDateSection = QLabel("Time Date Settings:")
        label_timeDateSection.setFont(boldFont)
        label_DateFormat = QLabel("Set Calendar Date Format\n"
                                  "(Needs to be the same with files):")
        label_DateDelimiter = QLabel("Delimiter:")

        # Set SpinBox Layouts
        hbox_startYear = QHBoxLayout()
        hbox_startYear.addWidget(label_StartYear)
        hbox_startYear.addWidget(self.spinBox_startYear)

        hbox_endYear = QHBoxLayout()
        hbox_endYear.addWidget(label_EndYear)
        hbox_endYear.addWidget(self.spinBox_endYear)

        hbox_yearLayout = QHBoxLayout()
        hbox_yearLayout.addLayout(hbox_startYear)
        hbox_yearLayout.addLayout(hbox_endYear)
        hbox_yearLayout.addSpacerItem(QSpacerItem(_INT_MAX_STRETCH, 0))

        vbox_finalYearLayout = QVBoxLayout()
        vbox_finalYearLayout.addWidget(label_yearSection)
        vbox_finalYearLayout.addLayout(hbox_yearLayout)

        # Set Radio Buttons
        buttGroup_DateDelim = QButtonGroup(self)
        buttGroup_DateDelim.addButton(self.radioButton_DateSlash)
        buttGroup_DateDelim.addButton(self.radioButton_DateDash)
        buttGroup_DateDelim.addButton(self.radioButton_DateCustom)

        hbox_dateButtons = QHBoxLayout()
        hbox_dateButtons.addWidget(label_DateDelimiter)
        hbox_dateButtons.addWidget(self.radioButton_DateSlash)
        hbox_dateButtons.addWidget(self.radioButton_DateDash)
        hbox_dateButtons.addWidget(self.radioButton_DateCustom)
        hbox_dateButtons.addWidget(self.lineEdit_DateCustom)

        # Set ComboBox
        hbox_ComboDate = QHBoxLayout()
        hbox_ComboDate.addWidget(label_DateFormat)
        hbox_ComboDate.addWidget(self.comboBox_DateFormat)
        hbox_ComboDate.addLayout(hbox_dateButtons)
        hbox_ComboDate.addSpacerItem(QSpacerItem(250, 0))

        vbox_finalTimeDateLayout = QVBoxLayout()
        vbox_finalTimeDateLayout.addWidget(label_timeDateSection)
        vbox_finalTimeDateLayout.addLayout(hbox_ComboDate)

        # Main Layout
        self.vbox_main_layout.addLayout(vbox_finalYearLayout)
        self.vbox_main_layout.addLayout(vbox_finalTimeDateLayout)
        self.vbox_main_layout.addSpacerItem(QSpacerItem(0, _INT_MAX_STRETCH))

    # ------------------- #
    # ----- Setters ----- #
    # ------------------- #
    def setComboBoxes(self):
        # DateFileFormat
        self.comboBox_DateFormat.addItem(my_cal_v2.DD_MM_YYYY)
        self.comboBox_DateFormat.addItem(my_cal_v2.DD_YYYY_MM)
        self.comboBox_DateFormat.addItem(my_cal_v2.D_M_YYYY)
        self.comboBox_DateFormat.addItem(my_cal_v2.D_YYYY_M)
        self.comboBox_DateFormat.addItem(my_cal_v2.DD_MM_YY)
        self.comboBox_DateFormat.addItem(my_cal_v2.DD_YY_MM)
        self.comboBox_DateFormat.addItem(my_cal_v2.D_M_YY)
        self.comboBox_DateFormat.addItem(my_cal_v2.D_YY_M)

        self.comboBox_DateFormat.addItem(my_cal_v2.MM_DD_YYYY)
        self.comboBox_DateFormat.addItem(my_cal_v2.MM_YYYY_DD)
        self.comboBox_DateFormat.addItem(my_cal_v2.M_D_YYYY)
        self.comboBox_DateFormat.addItem(my_cal_v2.M_YYYY_D)
        self.comboBox_DateFormat.addItem(my_cal_v2.MM_DD_YY)
        self.comboBox_DateFormat.addItem(my_cal_v2.MM_YY_DD)
        self.comboBox_DateFormat.addItem(my_cal_v2.M_D_YY)
        self.comboBox_DateFormat.addItem(my_cal_v2.M_YY_D)

        self.comboBox_DateFormat.addItem(my_cal_v2.YYYY_MM_DD)
        self.comboBox_DateFormat.addItem(my_cal_v2.YYYY_DD_MM)
        self.comboBox_DateFormat.addItem(my_cal_v2.YYYY_M_D)
        self.comboBox_DateFormat.addItem(my_cal_v2.YYYY_D_M)
        self.comboBox_DateFormat.addItem(my_cal_v2.YY_MM_DD)
        self.comboBox_DateFormat.addItem(my_cal_v2.YY_DD_MM)
        self.comboBox_DateFormat.addItem(my_cal_v2.YY_M_D)
        self.comboBox_DateFormat.addItem(my_cal_v2.YY_D_M)

        self.comboBox_DateFormat.setCurrentText(_MYCALV2_DEFAULT_DATE_FORMAT)

# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetMergeTableFilesCalendar(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                              winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
