import sys
import os
import tkinter as tk
import qdarkstyle
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QMainWindow, QApplication, QWidget
from PySide2.QtGui import QIcon

_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width


class MainWindowTemplate(QMainWindow):
    def __init__(self, app, w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath='', parent=None):
        super(MainWindowTemplate, self).__init__(parent)  # super().__init__()
        self.app = app

        # -------------------------- #
        # ----- Set MainWindow ----- #
        # -------------------------- #
        self.setStyle_()
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
        self.setGeometry(_INT_SCREEN_WIDTH / 4, _INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height

    # -------------------------- #
    # ----- Static Methods ----- #
    # -------------------------- #
    @staticmethod
    def widgetDialogParams(widget: QWidget):
        widget.setWindowModality(Qt.ApplicationModal)
        # widget.setWindowFlags(Qt.WindowStaysOnTopHint)

    @staticmethod
    def setSpaces(number):
        return number * ' '

    # ------------------------------ #
    # ----- Non-Static Methods ----- #
    # ------------------------------ #

    def setStyle_(self):
        self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    mainWin = MainWindowTemplate(myApp, w=w, h=h, minW=minW, minH=minH, winTitle=winTitle,
                                 iconPath=iconPath)  # Create MainWindow
    mainWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


# ****************************************************** #
# ********************   __main__   ******************** #
# ****************************************************** #
if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256,
             winTitle='SPACE', iconPath=_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
