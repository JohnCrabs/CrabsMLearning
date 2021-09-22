import sys
import os
import tkinter as tk
from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget, QListWidgetItem, QFileDialog, QLabel, QTabWidget
from PySide2.QtGui import QIcon, QPixmap

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
_DKEY_INPUT_LIST = 'input-list'
_DKEY_OUTPUT_LIST = 'output-list'


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
                                               _DKEY_OUTPUT_LIST: []
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

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_InputColumns = QListWidget()
        self.listWidget_InputColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_OutputColumns = QListWidget()
        self.listWidget_OutputColumns.setSelectionMode(QListWidget.ExtendedSelection)

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

        # Set Time vbox
        labelInputList = QLabel("Input Columns:\n" +
                                "(the columns that will be used as training/test/validation " +
                                "input\nfor the machine learning)")
        vbox_listInputColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listInputColumns.addWidget(labelInputList)  # Add Label
        vbox_listInputColumns.addWidget(self.listWidget_InputColumns)  # Add Column List
        vbox_listInputColumns.addLayout(hbox_listInputButtons)  # Add Layout

        # Set Date vbox
        labelOutputList = QLabel("Output Columns:\n" +
                                 "(the columns that will be used as training/test/validation " +
                                 "output\nfor the machine learning)")
        vbox_listOutputColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listOutputColumns.addWidget(labelOutputList)  # Add Label
        vbox_listOutputColumns.addWidget(self.listWidget_OutputColumns)  # Add Column List
        vbox_listOutputColumns.addLayout(hbox_listOutputButtons)  # Add Layout

        # Set ListWidget in hbox
        hbox_listWidget = QHBoxLayout()  # Create Horizontal Layout
        hbox_listWidget.addLayout(vbox_listInputColumns)  # Add vbox_Combine_1 layout
        hbox_listWidget.addLayout(vbox_listOutputColumns)  # Add vbox_Combine_2 layout

        self.vbox_main_layout.addLayout(hbox_listWidget)


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
