import sys
import os

from PySide2.QtCore import (
    Qt
)

from PySide2.QtOpenGL import (
    QGLWidget
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
    QPixmap,
    QColor
)

from lib.core.project_flags import *
# import lib.core.machineLearningRegression as mlr
# import lib.core.signalCompare as signComp

from lib.gui.guiStyle import setStyle_
import lib.gui.commonFunctions as coFunc


class WidgetMachineLearningImageClassification(QWidget):
    def __init__(self, w=512, h=512, minW=256, minH=256, maxW=512, maxH=512,
                 winTitle='My Window', iconPath=''):
        super().__init__()
        self.setStyleSheet(setStyle_())  # Set the styleSheet
        self.iconPath = iconPath

        # Set this flag to True to show debugging messages to console
        self.debugMessageFlag = False

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
        self.widgetTabInputOutput = WidgetTabImageVisualizer()  # create a tab for input output columns

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
        self.listWidget_FileList.setMaximumWidth(400)
        self.listWidget_ColumnList = QListWidget()  # Create a ListWidget
        self.listWidget_ColumnList.setMinimumWidth(300)
        self.listWidget_ColumnList.setMaximumWidth(400)
        self.listWidget_ColumnList.setSelectionMode(QListWidget.ExtendedSelection)  # Set Extended Selection
        self.fileName = None

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def set_IO_Tab(self):
        # Set main Tab Widget
        self.widgetTabInputOutput.setWidget()  # Set the Tab File Management Widget
        self.mainTabWidget.addTab(self.widgetTabInputOutput, "Image Visualizer")  # Add it to mainTanWidget

    def setWidget(self):
        """
        A function to create the widget components into the main QWidget
        :return: Nothing
        """
        self.set_IO_Tab()

        # Disable Generate Button
        self.buttonExecute.setEnabled(False)

        # Set Column vbox
        labelColumnList = QLabel("Sub-Folders (Classes) List:")
        vbox_listColumns = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listColumns.addWidget(labelColumnList)  # Add Label
        vbox_listColumns.addWidget(self.listWidget_ColumnList)  # Add Column List

        # Set add/remove button in vbox
        hbox_listFileButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listFileButtons.addWidget(self.buttonAdd)  # Add buttonAdd
        hbox_listFileButtons.addWidget(self.buttonRemove)  # Add buttonRemove
        hbox_listFileButtons.addWidget(self.buttonExecute)  # Add buttonGenerate

        # Set FileList in hbox
        labelFileList = QLabel("Opened Folders List:")
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

    # ------------------ #
    # ----- Events ----- #
    # ------------------ #
    # ***** SET EVENTS FUNCTIONS ***** #
    # ***** MAIN EVENTS ***** #
    def setMainEvents_(self):
        pass


# *********************************** #
# *********** Tab Widgets *********** #
# *********************************** #
# *                                 * #

# *********** Machine Learning I/O *********** #
class WidgetTabImageVisualizer(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet(setStyle_())  # set the tab style

        # ---------------------- #
        # ----- Set Window ----- #
        # ---------------------- #
        self.vbox_main_layout = QVBoxLayout(self)  # Create the main vbox

        # --------------------------- #
        # ----- Set QListWidget ----- #
        # --------------------------- #
        self.listWidget_ImageList = QListWidget()
        self.listWidget_ImageList.setMaximumHeight(256)

        # ------------------------------- #
        # ----- Set QGLWindow ----------- #
        # ------------------------------- #
        self.GLWindow_ImageVisualizer = OpenGLWidgetImageVisualizer()

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def setWidget(self):
        vbox_final = QVBoxLayout()
        vbox_final.addWidget(self.GLWindow_ImageVisualizer)
        vbox_final.addWidget(self.listWidget_ImageList)

        self.vbox_main_layout.addLayout(vbox_final)


# *********** OpenGL Image Visualizer *********** #
class OpenGLWidgetImageVisualizer(QGLWidget):
    def __init__(self, parent=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumWidth(256)
        self.setMinimumHeight(256)

    def initializeGL(self):
        self.qglClearColor(QColor('black'))

    def paintGL(self):
        self.qglClearColor(QColor('black'))

    def resizeGL(self, w: int, h: int):
        self.qglClearColor(QColor('black'))


# ******************************************************* #
# ********************   EXECUTION   ******************** #
# ******************************************************* #

def exec_app(w=512, h=512, minW=256, minH=256, maxW=512, maxH=512, winTitle='My Window', iconPath=''):
    myApp = QApplication(sys.argv)  # Set Up Application
    widgetWin = WidgetMachineLearningImageClassification(w=w, h=h, minW=minW, minH=minH, maxW=maxW, maxH=maxH,
                                                         winTitle=winTitle, iconPath=iconPath)  # Create MainWindow
    widgetWin.show()  # Show Window
    myApp.exec_()  # Execute Application
    sys.exit(0)  # Exit Application


if __name__ == "__main__":
    exec_app(w=1024, h=512, minW=512, minH=256, maxW=512, maxH=512,
             winTitle='WidgetTemplate', iconPath=PROJECT_FOLDER + '/icon/crabsMLearning_32x32.png')
