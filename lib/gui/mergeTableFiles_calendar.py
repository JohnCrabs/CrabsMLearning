import sys
import os
import tkinter as tk
from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget, QListWidgetItem, QFileDialog, QLabel
from PySide2.QtGui import QIcon, QPixmap

import lib.core.file_manipulation as file_manip

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


class WidgetMergeTableFilesCalendar(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()

        self.setStyle_()

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

        self.buttonGenerate = QPushButton("Generate")
        self.buttonGenerate.setMinimumWidth(0)  # Set Minimum Width
        self.buttonGenerate.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Height
        self.buttonGenerate.setToolTip('Open the generate window to proceed in the merging process.')  # Add Description

        self.buttonDateColumn = QPushButton("Date")
        self.buttonDateColumn.setMinimumWidth(0)  # Set Minimum Width
        self.buttonDateColumn.setMinimumHeight(0)  # Set Minimum Height
        self.buttonDateColumn.setToolTip('Set selected column as Date Column.')  # Add Description

        self.buttonRemDateColumn = QPushButton("Remove")
        self.buttonRemDateColumn.setMinimumWidth(0)  # Set Minimum Width
        self.buttonRemDateColumn.setMinimumHeight(0)  # Set Minimum Height
        self.buttonRemDateColumn.setToolTip('Remove selected column as Date Column.')  # Add Description

        self.buttonPrimaryEvent = QPushButton("Primary Event")
        self.buttonPrimaryEvent.setMinimumWidth(0)  # Set Minimum Width
        self.buttonPrimaryEvent.setMinimumHeight(0)  # Set Minimum Height
        self.buttonPrimaryEvent.setToolTip('Set selected column as Primary Event Column.')  # Add Description

        self.buttonRemPrimaryEvent = QPushButton("Remove")
        self.buttonRemPrimaryEvent.setMinimumWidth(0)  # Set Minimum Width
        self.buttonRemPrimaryEvent.setMinimumHeight(0)  # Set Minimum Height
        self.buttonRemPrimaryEvent.setToolTip('Remove selected column as Primary Event Column.')  # Add Description

        self.buttonEvent = QPushButton("Event Column")
        self.buttonEvent.setMinimumWidth(0)  # Set Minimum Width
        self.buttonEvent.setMinimumHeight(0)  # Set Minimum Height
        self.buttonEvent.setToolTip('Set selected column as Event Column.')  # Add Description

        self.buttonRemEvent = QPushButton("Remove")
        self.buttonRemEvent.setMinimumWidth(0)  # Set Minimum Width
        self.buttonRemEvent.setMinimumHeight(0)  # Set Minimum Height
        self.buttonRemEvent.setToolTip('Remove selected column as Event Column.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_FileList = QListWidget()

        self.listWidget_ColumnList = QListWidget()
        self.listWidget_ColumnList.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_DateColumns = QListWidget()
        self.listWidget_DateColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_PrimEventColumns = QListWidget()
        self.listWidget_PrimEventColumns.setSelectionMode(QListWidget.ExtendedSelection)
        self.listWidget_EventColumns = QListWidget()
        self.listWidget_EventColumns.setSelectionMode(QListWidget.ExtendedSelection)

        # ----------------------- #
        # ----- Set Actions ----- #
        # ----------------------- #
        self.setEvents_()

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #
        self.str_pathToTheProject = _NEW_PROJECT_DEFAULT_FOLDER  # var to store the projectPath
        self.dict_tableFilesPaths = {}  # a dictionary to store the table files
        self.dict_DateColumns = {}  # a dictionary to store the date columns
        self.dict_PrimEventColumns = {}  # a dictionary to store primary event columns
        self.dict_EventColumns = {}  # a dictionary to store the merging event columns

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #

    def setStyle_(self):
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
                """
        self.setStyleSheet(style)

    def setWidget(self):
        # Set add/remove button in vbox
        hbox_listFileButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listFileButtons.addWidget(self.buttonAdd)  # Add buttonAdd
        hbox_listFileButtons.addWidget(self.buttonRemove)  # Add buttonRemove
        hbox_listFileButtons.addWidget(self.buttonGenerate)

        # Set FileList in hbox
        labelFileList = QLabel("Opened File List:")
        vbox_listFile = QVBoxLayout()  # Create a Vertical Box Layout
        vbox_listFile.addWidget(labelFileList)  # Add Label
        vbox_listFile.addWidget(self.listWidget_FileList)  # Add FileList
        vbox_listFile.addLayout(hbox_listFileButtons)  # Add vbox_listFileButtons layout

        # Set column/remove buttons in vbox
        hbox_listDateButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        # hbox_listDateButtons.addSpacerItem(QSpacerItem(_INT_MAX_STRETCH, 0))
        hbox_listDateButtons.addWidget(self.buttonDateColumn)  # Add buttonDate
        hbox_listDateButtons.addWidget(self.buttonRemDateColumn)  # Add buttonRemove

        hbox_listPrimEventButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listPrimEventButtons.addWidget(self.buttonPrimaryEvent)  # Add buttonDate
        hbox_listPrimEventButtons.addWidget(self.buttonRemPrimaryEvent)  # Add buttonRemove

        hbox_listEventsButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listEventsButtons.addWidget(self.buttonEvent)  # Add buttonDate
        hbox_listEventsButtons.addWidget(self.buttonRemEvent)  # Add buttonRemove

        # Set Column vbox
        labelColumnList = QLabel("Column List:")
        vbox_listColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listColumns.addWidget(labelColumnList)  # Add Label
        vbox_listColumns.addWidget(self.listWidget_ColumnList)  # Add Column List

        # Set Date vbox
        labelDateList = QLabel("Date Column (one common column for each file):")
        vbox_listDateColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listDateColumns.addWidget(labelDateList)  # Add Label
        vbox_listDateColumns.addWidget(self.listWidget_DateColumns)  # Add Column List
        vbox_listDateColumns.addLayout(hbox_listDateButtons)

        # Set PrimEvent vbox
        labelPrimEventList = QLabel("Primary Event (one common column for each file):")
        vbox_listPrimEventColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listPrimEventColumns.addWidget(labelPrimEventList)  # Add Label
        vbox_listPrimEventColumns.addWidget(self.listWidget_PrimEventColumns)  # Add Column List
        vbox_listPrimEventColumns.addLayout(hbox_listPrimEventButtons)

        # Set EventColumns vbox
        labelEventColumnsList = QLabel("Other Events to be merged (they will be set under primary event):")
        vbox_listEventColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listEventColumns.addWidget(labelEventColumnsList)  # Add Label
        vbox_listEventColumns.addWidget(self.listWidget_EventColumns)  # Add Column List
        vbox_listEventColumns.addLayout(hbox_listEventsButtons)

        # Combine Column Boxes vbox
        vbox_Combine_1 = QVBoxLayout()
        vbox_Combine_1.addLayout(vbox_listColumns)
        vbox_Combine_1.addLayout(vbox_listPrimEventColumns)

        vbox_Combine_2 = QVBoxLayout()
        vbox_Combine_2.addLayout(vbox_listDateColumns)
        vbox_Combine_2.addLayout(vbox_listEventColumns)

        # Set ListWidget in hbox
        hbox_listWidget = QHBoxLayout()  # Create Horizontal Layout
        hbox_listWidget.addLayout(vbox_listFile)  # Add vbox_listFile layout
        hbox_listWidget.addLayout(vbox_Combine_1)  # Add vbox_Combine_1 layout
        hbox_listWidget.addLayout(vbox_Combine_2)  # Add vbox_Combine_2 layout

        self.vbox_main_layout.addLayout(hbox_listWidget)

    def openFileDialog(self, dialogName='Pick a File', dialogOpenAt=file_manip.PATH_HOME, dialogFilters=None,
                       dialogMultipleSelection: bool = False):
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

    def addItemsToList(self, fullPath):
        fileName = fullPath.split('/')[-1:][0]
        self.dict_tableFilesPaths[fileName] = {'name': fileName,
                                               'full_path': fullPath,
                                               'columns': file_manip.getColumnNames(fullPath, splitter=',')
                                               }
        self.resetDateColumn(fileName)
        self.resetPrimEventColumn(fileName)
        self.dict_EventColumns[fileName] = []
        self.listWidget_FileList.addItem(QListWidgetItem(fileName))  # Add Item to List

    def updateDateList(self):
        self.listWidget_DateColumns.clear()
        for key in self.dict_DateColumns.keys():
            if self.dict_DateColumns[key] is not None:
                self.listWidget_DateColumns.addItem(QListWidgetItem(key + " -> " + self.dict_DateColumns[key]))

    def updatePrimaryEventList(self):
        self.listWidget_PrimEventColumns.clear()
        for key in self.dict_PrimEventColumns.keys():
            if self.dict_PrimEventColumns[key] is not None:
                self.listWidget_PrimEventColumns.addItem(QListWidgetItem(key + " -> " + self.dict_PrimEventColumns[key]))

    def updateEventsList(self):
        self.listWidget_EventColumns.clear()
        for key in self.dict_EventColumns.keys():
            if self.dict_EventColumns[key] is not []:
                for event in self.dict_EventColumns[key]:
                    self.listWidget_EventColumns.addItem(QListWidgetItem(key + " -> " + event))

    def resetDateColumn(self, fileName):
        self.dict_DateColumns[fileName] = None

    def resetPrimEventColumn(self, fileName):
        self.dict_PrimEventColumns[fileName] = None

    def resetEventColumn(self, fileName):
        self.dict_EventColumns[fileName] = []

    def removeFromEventColumn(self, fileName, column):
        if column in self.dict_EventColumns[fileName]:
            self.dict_EventColumns[fileName].remove(column)

    # -------------------------------- #
    # ----- Print/Show Functions ----- #
    # -------------------------------- #
    def prt_dict_tableFilePaths(self):
        for key in self.dict_tableFilesPaths.keys():
            print("file-key: ", key)
            for sec_key in self.dict_tableFilesPaths[key].keys():
                print(str(sec_key) + ': ', self.dict_tableFilesPaths[key][sec_key])
                print()
            print()

    # ------------------ #
    # ----- Events ----- #
    # ------------------ #
    def setEvents_(self):
        # Button Events
        self.buttonAdd.clicked.connect(self.actionButtonAdd)
        self.buttonRemove.clicked.connect(self.actionButtonRemove)

        self.buttonDateColumn.clicked.connect(self.actionButtonDateColumn)
        self.buttonRemDateColumn.clicked.connect(self.actionButtonRemDateColumn)

        self.buttonPrimaryEvent.clicked.connect(self.actionButtonPrimaryEvent)
        self.buttonRemPrimaryEvent.clicked.connect(self.actionButtonRemPrimaryEvent)

        self.buttonEvent.clicked.connect(self.actionButtonEvent)
        self.buttonRemEvent.clicked.connect(self.actionButtonRemEvent)

        # ListWidget Events
        self.listWidget_FileList.currentRowChanged.connect(self.fileListRowChanged_event)

    def actionButtonAdd(self):
        # Open a dialog for CSV files
        success, dialog = self.openFileDialog(dialogName='Open Table File (Currently strictly CSV)',
                                              dialogOpenAt=self.str_pathToTheProject,
                                              dialogFilters=["CSV File Format (*.csv)"],
                                              dialogMultipleSelection=True)

        if success:  # if True
            for filePath in dialog.selectedFiles():  # for each file in all selected files
                fileName = filePath.split('/')[-1:][0]
                # print(fullPath)
                # print(fileName)
                # print()
                if fileName not in self.dict_tableFilesPaths.keys():
                    self.addItemsToList(filePath)

            if self.listWidget_FileList.currentItem() is None:  # Set row 0 as current row
                self.listWidget_FileList.setCurrentRow(0)  # Set current row
            # self.prt_dict_tableFilePaths()

    def actionButtonRemove(self):
        if self.listWidget_FileList.currentItem() is not None:
            self.dict_tableFilesPaths.pop(self.listWidget_FileList.currentItem().text(), None)
            self.dict_DateColumns.pop(self.listWidget_FileList.currentItem().text(), None)
            self.dict_PrimEventColumns.pop(self.listWidget_FileList.currentItem().text(), None)
            self.dict_EventColumns.pop(self.listWidget_FileList.currentItem().text(), None)
            self.listWidget_FileList.takeItem(self.listWidget_FileList.currentRow())
            self.fileListRowChanged_event()
            self.updateDateList()
            self.updatePrimaryEventList()
            self.updateEventsList()

    def actionButtonDateColumn(self):
        if self.listWidget_FileList.currentItem() is not None and self.listWidget_ColumnList.currentItem() is not None:
            currentFileName = self.listWidget_FileList.currentItem().text()
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            # print(currentFileName, " -> ", currentColumnSelected)
            if self.dict_PrimEventColumns[currentFileName] == currentColumnSelected:
                self.resetPrimEventColumn(currentFileName)
            elif currentColumnSelected in self.dict_EventColumns[currentFileName]:
                self.removeFromEventColumn(currentFileName, currentColumnSelected)
            self.dict_DateColumns[currentFileName] = currentColumnSelected
            self.updatePrimaryEventList()
            self.updateEventsList()
            self.updateDateList()

    def actionButtonRemDateColumn(self):
        if self.listWidget_DateColumns.currentItem() is not None:
            selectedItems = self.listWidget_DateColumns.selectedItems()
            for item in selectedItems:
                tmp_str = item.text()
                fileName = tmp_str.split(' -> ')[0]
                self.resetDateColumn(fileName)
            self.updateDateList()

    def actionButtonPrimaryEvent(self):
        if self.listWidget_FileList.currentItem() is not None and self.listWidget_ColumnList.currentItem() is not None:
            currentFileName = self.listWidget_FileList.currentItem().text()
            currentColumnSelected = self.listWidget_ColumnList.currentItem().text()
            if self.dict_DateColumns[currentFileName] == currentColumnSelected:
                self.resetDateColumn(currentFileName)
            elif currentColumnSelected in self.dict_EventColumns[currentFileName]:
                self.removeFromEventColumn(currentFileName, currentColumnSelected)
            # print(currentFileName, " -> ", currentColumnSelected)
            self.dict_PrimEventColumns[currentFileName] = currentColumnSelected
            self.updateDateList()
            self.updateEventsList()
            self.updatePrimaryEventList()

    def actionButtonRemPrimaryEvent(self):
        if self.listWidget_PrimEventColumns.currentItem() is not None:
            selectedItems = self.listWidget_PrimEventColumns.selectedItems()
            for item in selectedItems:
                tmp_str = item.text()
                fileName = tmp_str.split(' -> ')[0]
                self.resetPrimEventColumn(fileName)
            self.updatePrimaryEventList()

    def actionButtonEvent(self):
        if self.listWidget_FileList.currentItem() is not None and self.listWidget_ColumnList.currentItem() is not None:
            currentFileName = self.listWidget_FileList.currentItem().text()
            currentSelectedItems = self.listWidget_ColumnList.selectedItems()
            for currentColumnSelected in currentSelectedItems:
                # print(currentFileName, " -> ", currentColumnSelected.text())
                if self.dict_DateColumns[currentFileName] == currentColumnSelected.text():
                    self.resetDateColumn(currentFileName)
                elif self.dict_PrimEventColumns[currentFileName] == currentColumnSelected.text():
                    self.resetPrimEventColumn(currentFileName)
                if currentColumnSelected.text() not in self.dict_EventColumns[currentFileName]:
                    self.dict_EventColumns[currentFileName].append(currentColumnSelected.text())
            self.updateDateList()
            self.updatePrimaryEventList()
            self.updateEventsList()

    def actionButtonRemEvent(self):
        if self.listWidget_EventColumns.currentItem() is not None:
            selectedItems = self.listWidget_EventColumns.selectedItems()
            for item in selectedItems:
                tmp_str = item.text()
                fileName = tmp_str.split(' -> ')[0]
                columnName = tmp_str.split(' -> ')[1]
                self.removeFromEventColumn(fileName, columnName)
            self.updateEventsList()

    def fileListRowChanged_event(self):
        self.listWidget_ColumnList.clear()
        if self.listWidget_FileList.currentItem() is not None:
            fileName = self.listWidget_FileList.currentItem().text()
            for column in self.dict_tableFilesPaths[fileName]['columns']:
                self.listWidget_ColumnList.addItem(QListWidgetItem(column))

            if self.listWidget_ColumnList.currentItem() is None:  # Set first row selected
                self.listWidget_ColumnList.setCurrentRow(0)


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
