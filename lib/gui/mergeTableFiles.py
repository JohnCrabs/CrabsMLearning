import sys
import os
import tkinter as tk
from PySide2.QtCore import QUrl
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget, QFileDialog
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


class WidgetMergeTableFiles(QWidget):
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
        self.buttonAdd.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonAdd.setMaximumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonAdd.setIcon(QIcon(QPixmap(_ICON_ADD)))  # Add Icon
        self.buttonAdd.setToolTip('Add table files.')  # Add Description

        self.buttonRemove = QPushButton()  # Create button for Remove
        self.buttonRemove.setMinimumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonRemove.setMaximumWidth(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonRemove.setMinimumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Minimum Width
        self.buttonRemove.setMaximumHeight(_INT_ADD_REMOVE_BUTTON_SIZE)  # Set Maximum Width
        self.buttonRemove.setIcon(QIcon(QPixmap(_ICON_REMOVE)))  # Add Icon
        self.buttonAdd.setToolTip('Remove table files.')  # Add Description

        # -------------------------------- #
        # ----- Set QListWidgetItems ----- #
        # -------------------------------- #
        self.listWidget_FileList = QListWidget()

        self.listWidget_ColumnList = QListWidget()

        # ----------------------- #
        # ----- Set Actions ----- #
        # ----------------------- #
        self.setActions_()

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #
        self.str_pathToTheProject = _NEW_PROJECT_DEFAULT_FOLDER  # var to store the projectPath
        self.dict_tableFilesPaths = {}  # a dictionary to store the table files

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #

    def setStyle_(self):
        style = """
                QListWidget {
                    background-color: white;
                }
                
                QPushButton {
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

        # Set FileList in hbox
        vbox_listFile = QVBoxLayout()  # Create a Vertical Box Layout
        vbox_listFile.addWidget(self.listWidget_FileList)  # Add FileList
        vbox_listFile.addLayout(hbox_listFileButtons)  # Add vbox_listFileButtons layout

        # Set ListWidget in hbox
        hbox_listWidget = QHBoxLayout()  # Create Horizontal Layout
        hbox_listWidget.addLayout(vbox_listFile)  # Add hbox_listFile layout
        hbox_listWidget.addWidget(self.listWidget_ColumnList)  # Add ColumnList

        # hbox_buttons.addSpacerItem(QSpacerItem(_INT_MAX_STRETCH, 0))  # Add Spacer
        # hbox_buttons.addWidget(self.buttonOk)  # Add the OK Button
        # hbox_buttons.addWidget(self.buttonCancel)  # Add the CANCEL Button

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

    # -------------------------------- #
    # ----- Print/Show Functions ----- #
    # -------------------------------- #
    def prt_dict_tableFilePaths(self):
        for key in self.dict_tableFilesPaths.keys():
            print("file-key: " + key)
            for sec_key in self.dict_tableFilesPaths[key].keys():
                print(str(sec_key) + ': ')
                print(self.dict_tableFilesPaths[key][sec_key])
            print()

    # ------------------- #
    # ----- Actions ----- #
    # ------------------- #
    def setActions_(self):
        self.buttonAdd.clicked.connect(self.actionButtonAdd)
        self.buttonRemove.clicked.connect(self.actionButtonRemove)

    def actionButtonAdd(self):
        # Open a dialog for CSV files
        success, dialog = self.openFileDialog(dialogName='Open Table File (Currently strictly CSV)',
                                              dialogOpenAt=self.str_pathToTheProject,
                                              dialogFilters=["CSV File Format (*.csv)"],
                                              dialogMultipleSelection=True)

        if success:  # if True
            for filePath in dialog.selectedFiles():  # for each file in all selected files
                fullPath = filePath
                fileName = filePath.split('/')[-1:][0]
                # print(fullPath)
                # print(fileName)
                # print()
                self.dict_tableFilesPaths[fileName] = {'name': fileName,
                                                       'full_path': fullPath,
                                                       'columns': file_manip.getColumnNames(fullPath, splitter=','),
                                                       'common_columns': '',
                                                       'merge_columns': ''}

            self.prt_dict_tableFilePaths()

    def actionButtonRemove(self):
        pass
        # self.close()  # Close the window


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #

def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetMergeTableFiles(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                      winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
