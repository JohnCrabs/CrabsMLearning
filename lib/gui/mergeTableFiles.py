import sys
import os
import tkinter as tk
from PySide2.QtWidgets import QWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QSpacerItem, \
    QListWidget
from PySide2.QtGui import QIcon, QPixmap

_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width

_ICON_ADD = _PROJECT_FOLDER + "/icon/add_cross_128x128_filled.png"
_ICON_REMOVE = _PROJECT_FOLDER + "/icon/remove_line_128x128_filled.png"


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
        self.buttonAdd.setIcon(QIcon(QPixmap(_ICON_ADD)))  # Add Icon
        self.buttonAdd.setMinimumWidth(64)  # Set Minimum Width
        self.buttonAdd.setMaximumWidth(64)  # Set Maximum Width
        self.buttonAdd.setToolTip('Add table files.')  # Add Description

        self.buttonRemove = QPushButton()  # Create button for Remove
        self.buttonRemove.setIcon(QIcon(QPixmap(_ICON_REMOVE)))  # Add Icon
        self.buttonRemove.setMinimumWidth(64)  # Set Minimum Width
        self.buttonRemove.setMaximumWidth(64)  # Set Maximum Width
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

    def setStyle_(self):
        style = """
                QListWidget {
                    background: lightgrey;
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

    # ------------------- #
    # ----- Actions ----- #
    # ------------------- #
    def setActions_(self):
        self.buttonAdd.clicked.connect(self.actionButtonAdd)
        self.buttonRemove.clicked.connect(self.actionButtonRemove)

    def actionButtonAdd(self):
        # -----> Write here code for ok <-----
        # self.close()  # Close the window
        pass

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
