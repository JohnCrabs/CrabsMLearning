import sys
import os
import tkinter as tk
import qdarkstyle
from PySide2.QtCore import Qt
from PySide2.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QStatusBar
from PySide2.QtGui import QIcon

from lib.gui.mergeTableFiles_simple import WidgetMergeTableFilesSimple
from lib.gui.mergeTableFiles_calendar import WidgetMergeTableFilesCalendar

from lib.gui.machineLearningRegressionWidget import WidgetMachineLearningRegressionWidget
from lib.gui.machineLearningImageClassification import WidgetMachineLearningImageClassification

_STR_PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

_INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
_INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
_INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
_INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

_INT_MAX_STRETCH = 100000  # Spacer Max Stretch
_INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width
_INT_SPACES = 10  # Set Spaces for Menu Items

# Icon Paths
_ICON_PATH_LOGO_32x32 = _STR_PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png'
_ICON_PATH_OPEN_128x128 = _STR_PROJECT_FOLDER + '/icon/open_128x128.png'
_ICON_PATH_SETTINGS_48x48 = _STR_PROJECT_FOLDER + '/icon/settings_48_48.png'
_ICON_PATH_EXIT_APP_48x48 = _STR_PROJECT_FOLDER + '/icon/exit_app_48x48.png'
_ICON_PATH_CALENDAR_48x48 = _STR_PROJECT_FOLDER + '/icon/calendar_48x48.png'


class MainWindowCrabsMLearning(QMainWindow):
    def __init__(self, app, w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath='', parent=None):
        super(MainWindowCrabsMLearning, self).__init__(parent)  # super().__init__()
        self.app = app

        # ----------------------------- #
        # ----- Set Other Widgets ----- #
        # ----------------------------- #

        # ***************************** #
        # Tools -> ....                 #

        # .... -> MergeTableFilesSimple
        self.widgetMergeTableFilesSimple = WidgetMergeTableFilesSimple(w=512, h=512,
                                                                       minW=512, minH=256,
                                                                       maxW=840, maxH=512,
                                                                       winTitle='Merge Table Files',
                                                                       iconPath=_ICON_PATH_LOGO_32x32)
        self.widgetMergeTableFilesSimple.setWidget()

        # .... -> MergeTableFilesCalendar
        self.widgetMergeTableFilesCalendar = WidgetMergeTableFilesCalendar(w=1024, h=512,
                                                                           minW=512, minH=256,
                                                                           maxW=1280, maxH=840,
                                                                           winTitle='Merge Table Files',
                                                                           iconPath=_ICON_PATH_LOGO_32x32)
        self.widgetMergeTableFilesCalendar.setWidget()

        # .... -> MachineLearningRegression
        self.widgetMachineLearningRegressionWidget = WidgetMachineLearningRegressionWidget(w=1024, h=512,
                                                                                           minW=512, minH=256,
                                                                                           maxW=None,
                                                                                           maxH=None,
                                                                                           winTitle='Machine Learning Regression',
                                                                                           iconPath=_ICON_PATH_LOGO_32x32)
        self.widgetMachineLearningRegressionWidget.setWidget()

        # .... -> MachineLearningForVideo
        self.widgetMachineLearningForVideoWidget = WidgetMachineLearningImageClassification(w=1024, h=512,
                                                                                            minW=512, minH=256,
                                                                                            maxW=None,
                                                                                            maxH=None,
                                                                                            winTitle='Machine Learning Image Classification',
                                                                                            iconPath=_ICON_PATH_LOGO_32x32)
        self.widgetMachineLearningForVideoWidget.setWidget()

        #                               #
        # ***************************** #

        # -------------------------- #
        # ----- Set MainWindow ----- #
        # -------------------------- #
        self.setStyle_()
        self.setWindowTitle(winTitle)  # Set Window Title
        self.setWindowIcon(QIcon(iconPath))  # Set Window Icon
        self.setGeometry(_INT_SCREEN_WIDTH / 4, _INT_SCREEN_HEIGHT / 4, w, h)  # Set Window Geometry
        self.setMinimumWidth(minW)  # Set Window Minimum Width
        self.setMinimumHeight(minH)  # Set Window Minimum Height

        # ----------------------- #
        # ----- Set MenuBar ----- #
        # ----------------------- #
        self.mainMenu = self.menuBar()  # Set the Menu Bar

        # ***** ACTIONS ***** #
        # ____ FILE ____ #
        self.actionExit = QAction(QIcon(_ICON_PATH_EXIT_APP_48x48), 'Exit' + self.setSpaces(_INT_SPACES))  # Exit
        self.actionExit.setShortcut('Ctrl+Q')  # Ctrl + Q
        self.actionExit.setToolTip('Application exit.')  # ToolTip

        # ____ TOOLS ____ #
        # Action MergeTF_Simple
        self.actionMergeTF_Simple = QAction('Simple Merge' + self.setSpaces(_INT_SPACES))  # MergeTableFiles
        # self.actionMergeTableFiles.setShortcut()
        self.actionMergeTF_Simple.setToolTip('Merge multiple table files using specific columns in one table file.')

        # Action MergeTF_Calendar
        self.actionMergeTF_Calendar = QAction(QIcon(_ICON_PATH_CALENDAR_48x48), 'Calendar Merge' +
                                              self.setSpaces(_INT_SPACES))  # MergeTableFiles
        # self.actionMergeTableFiles.setShortcut()
        self.actionMergeTF_Calendar.setToolTip('Merge multiple table files using specific columns in one table file.' +
                                               ' One at least must be date column.')

        # Action MachineLearningRegression - ?? Machine Learning Regression Widget
        self.actionMachineLearningRegression = QAction('Machine Learning Regression' +
                                                       self.setSpaces(_INT_SPACES))

        # Action MachineLearningForVideo - ?? Machine Learning Widget for video
        self.actionMachineLearningForVideo = QAction('Machine Learning For Image Classification' +
                                                     self.setSpaces(_INT_SPACES))

        # ******************* #

        self.createMenuBar()  # Create all Menu/Sub-Menu/Actions

        # ---------------------------- #
        # ----- Set Main Content ----- #
        # ---------------------------- #

        # ------------------------- #
        # ----- Set StatusBar ----- #
        # ------------------------- #
        self.statusBar = QStatusBar()  # Create Status Bar

        # ------------------------------- #
        # ----- Set Actions Signals ----- #
        # ------------------------------- #
        self.setEvents_SignalSlots()  # Contains all the actions

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #

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

    # ---------------------------- #
    # ----- Override Methods ----- #
    # ---------------------------- #

    def closeEvent(self, event):
        self.actionExit_func_()

    # ------------------------------ #
    # ----- Non-Static Methods ----- #
    # ------------------------------ #

    def setStyle_(self):
        self.app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside2'))

    def createMenuBar(self):
        """
        This function runs all the createMenuBar* menu functions to create each menu.
        By default the template have a Menu File and a Menu Tool
        :return: Nothing
        """
        # Create Menu
        self.createMenuBarFile()  # File
        self.createMenuBarTools()  # Tools

    def createMenuBarFile(self):
        """
        Use this function to create the Menu File.
        Useful Commands:
        menuMain = self.mainMenu.addMenu('NewMenuName')
        menuMain.addAction(self.Action)  # add action created in def __init__()
        menuMain.addSeparator()  # add a separator line between Actions/Menus
        menuNewMenu = menuMain.addMenu("NewMenu")  # create a new Menu inside menuMain
        :return: Nothing
        """
        menuFile = self.mainMenu.addMenu('File')  # File
        # Set Actions and Menus to menuFile
        # Project Actions (New Project, Open, Save, etc)
        menuFile.addSeparator()  # Separator
        # Action Exit
        menuFile.addAction(self.actionExit)

    def createMenuBarTools(self):
        """
        Use this function to create the Menu File.
        Useful Commands:
        menuMain = self.mainMenu.addMenu('NewMenuName')
        menuMain.addAction(self.Action)  # add action created in def __init__()
        menuMain.addSeparator()  # add a separator line between Actions/Menus
        menuNewMenu = menuMain.addMenu("NewMenu")  # create a new Menu inside menuMain
        :return: Nothing
        """
        menuTools = self.mainMenu.addMenu('Tools')  # File
        # Set Actions and Menus to menuTools
        # Project Menu/Actions (Calendar, Machine Learning, )

        menuMergeTableFiles = menuTools.addMenu("Merge Table Files")
        menuMergeTableFiles.addAction(self.actionMergeTF_Simple)
        menuMergeTableFiles.addAction(self.actionMergeTF_Calendar)

        menuTools.addSeparator()

        menuMachineLearning = menuTools.addMenu("Machine Learning")
        menuMachineLearning.addAction(self.actionMachineLearningRegression)
        menuMachineLearning.addAction(self.actionMachineLearningForVideo)

    # ------------------ #
    # ----- Events ----- #
    # ------------------ #
    def setEvents_SignalSlots(self):
        """
        A function for storing all the trigger connections
        :return: Nothing
        """
        # ----------------- #
        # Triggered Actions #
        # ----------------- #
        # ********* #
        # Menu FILE #
        # ********* #
        self.actionExit.triggered.connect(self.actionExit_func_)  # actionExit

        # ********** #
        # Menu TOOLS #
        # ********** #
        self.actionMergeTF_Simple.triggered.connect(self.actionMergeTF_Simple_func_)  # actionMergeTF_Simple
        self.actionMergeTF_Calendar.triggered.connect(self.actionMergeTF_Calendar_func_)  # actionMergeTF_Calendar

        # actionMachineLearningSequential
        self.actionMachineLearningRegression.triggered.connect(self.actionMachineLearningRegression_func_)
        self.actionMachineLearningForVideo.triggered.connect(self.actionMachineLearningForVideo_func_)

    # ************ #
    # *** File *** #
    # ************ #
    def actionExit_func_(self):
        self.close()  # close the application
        QApplication.closeAllWindows()

    # ************* #
    # *** Tools *** #
    # ************* #
    def actionMergeTF_Simple_func_(self):
        self.widgetMergeTableFilesSimple.show()

    def actionMergeTF_Calendar_func_(self):
        self.widgetMergeTableFilesCalendar.show()

    def actionMachineLearningSequential_func_(self):
        self.widgetMachineLearningSequential.show()

    def actionMachineLearningMean_func_(self):
        self.widgetMachineLearningMean.show()

    def actionMachineLearningRegression_func_(self):
        self.widgetMachineLearningRegressionWidget.show()

    def actionMachineLearningForVideo_func_(self):
        self.widgetMachineLearningForVideoWidget.show()


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #


def exec_app(w=512, h=512, minW=256, minH=256, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    mainWin = MainWindowCrabsMLearning(myApp, w=w, h=h, minW=minW, minH=minH, winTitle=winTitle,
                                       iconPath=iconPath)  # Create MainWindow
    mainWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


# ****************************************************** #
# ********************   __main__   ******************** #
# ****************************************************** #
if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256,
             winTitle='CrabsMLearning', iconPath=_ICON_PATH_LOGO_32x32)
