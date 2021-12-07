import sys
import cv2 as cv2
from OpenGL.GL import *

from PySide2.QtCore import (
    QBasicTimer
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
    QListWidget,
    QListWidgetItem,
    # QFileDialog,
    QLabel,
    QTabWidget,
)

from PySide2.QtGui import (
    QIcon,
    QPixmap
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
        self._DKEY_DIR_NAME: str = 'name'
        self._DKEY_FULLPATH: str = 'full-path'
        self._DKEY_CLASSES_NAMES: str = 'classes-names'
        self._DKEY_CLASSES_DATA: str = 'classes-data'
        self._DKEY_IMAGE_NAMES: str = 'image-names'
        self._DKEY_DATA_FULLPATHS: str = 'data-fullpaths'
        self._DKEY_DATA_SIZE: str = 'data-size'

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
        self.widgetTabImageVisualizer = WidgetTabImageVisualizer()  # create a tab for input output columns

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
        self.listWidget_DirList = QListWidget()  # Create a ListWidget
        self.listWidget_DirList.setMinimumWidth(300)
        self.listWidget_DirList.setMaximumWidth(400)
        self.listWidget_ClassesList = QListWidget()  # Create a ListWidget
        self.listWidget_ClassesList.setMinimumWidth(300)
        self.listWidget_ClassesList.setMaximumWidth(400)
        self.listWidget_ClassesList.setSelectionMode(QListWidget.ExtendedSelection)  # Set Extended Selection
        self.fileName = None

        # --------------------- #
        # ----- Variables ----- #
        # --------------------- #
        self.str_pathToTheProject = NEW_PROJECT_DEFAULT_FOLDER  # var to store the projectPath
        self.dict_tableDirsPaths = {}  # a dictionary to store the table files

    # DICTIONARY FILE PARAMETERS
    def dkeyDirName(self):
        return self._DKEY_DIR_NAME

    def dkeyFullPath(self):
        return self._DKEY_FULLPATH

    def dkeyClassesNames(self):
        return self._DKEY_CLASSES_NAMES

    def dkeyClassesData(self):
        return self._DKEY_CLASSES_DATA

    def dkeyImageNames(self):
        return self._DKEY_IMAGE_NAMES

    def dkeyDataSize(self):
        return self._DKEY_DATA_SIZE

    def dkeyDataFullpath(self):
        return self._DKEY_DATA_FULLPATHS

    # --------------------------- #
    # ----- Reuse Functions ----- #
    # --------------------------- #
    def set_IO_Tab(self):
        # Set main Tab Widget
        self.widgetTabImageVisualizer.setWidget()  # Set the Tab File Management Widget
        self.mainTabWidget.addTab(self.widgetTabImageVisualizer, "Image Visualizer")  # Add it to mainTanWidget

    def setWidget(self):
        """
        A function to create the widget components into the main QWidget
        :return: Nothing
        """
        self.set_IO_Tab()

        # Disable Generate Button
        self.buttonExecute.setEnabled(False)

        # Set Classes vbox
        labelClassesList = QLabel("Sub-Folders (Classes) List:")
        vbox_listClasses = QVBoxLayout()  # Create a Horizontal Box Layout
        vbox_listClasses.addWidget(labelClassesList)  # Add Label
        vbox_listClasses.addWidget(self.listWidget_ClassesList)  # Add Classes List

        # Set add/remove button in vbox
        hbox_listFileButtons = QHBoxLayout()  # Create a Horizontal Box Layout
        hbox_listFileButtons.addWidget(self.buttonAdd)  # Add buttonAdd
        hbox_listFileButtons.addWidget(self.buttonRemove)  # Add buttonRemove
        hbox_listFileButtons.addWidget(self.buttonExecute)  # Add buttonGenerate

        # Set FileList in hbox
        labelFileList = QLabel("Opened Folders List:")
        vbox_listFile = QVBoxLayout()  # Create a Vertical Box Layout
        vbox_listFile.addWidget(labelFileList)  # Add Label
        vbox_listFile.addWidget(self.listWidget_DirList)  # Add FileList
        vbox_listFile.addLayout(vbox_listClasses)  # Add listClasses
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
        self.setImageVisualiserEvents()  # Set the events/action of ImageVisualiser

    # ---------------------------------- #
    # ----- Reuse Action Functions ----- #
    # ---------------------------------- #

    def addItemsToList(self, fullPath):
        dirName = fullPath.split('/')[-1:][0]  # find the name of the file
        dirClasses = os.listdir(fullPath)
        # Create the dictionary
        self.dict_tableDirsPaths[dirName] = {
            self.dkeyDirName(): dirName,
            self.dkeyFullPath(): fullPath,
            self.dkeyClassesNames(): dirClasses,
            self.dkeyClassesData(): {}
        }
        for _class_ in dirClasses:
            c_fullPath = os.path.normpath(fullPath + '/' + _class_)
            c_dataFileNames = os.listdir(c_fullPath)
            self.dict_tableDirsPaths[dirName][self.dkeyClassesData()][_class_] = {
                self.dkeyFullPath(): c_fullPath,
                self.dkeyImageNames(): c_dataFileNames,
                self.dkeyDataSize(): c_dataFileNames.__len__(),
                self.dkeyDataFullpath(): [os.path.normpath(c_fullPath + '/' + _fileName_) for _fileName_ in
                                          c_dataFileNames]
            }

        self.listWidget_DirList.addItem(QListWidgetItem(dirName))
        self.listWidget_DirList.setCurrentRow(0)
        self.listWidget_ClassesList.setCurrentRow(0)

    # ------------------ #
    # ----- Events ----- #
    # ------------------ #
    # ***** SET EVENTS FUNCTIONS ***** #
    # ***** MAIN EVENTS ***** #
    def setMainEvents_(self):
        # Button Events
        self.buttonAdd.clicked.connect(self.actionButtonAdd)  # buttonAdd -> clicked
        self.buttonRemove.clicked.connect(self.actionButtonRemove)  # buttonRemove -> clicked

        # ListWidget Events
        self.listWidget_DirList.currentRowChanged.connect(self.actionDirListRowChanged_event)
        self.listWidget_ClassesList.currentRowChanged.connect(self.actionClassesListRowChanged_event)

    def setImageVisualiserEvents(self):
        self.widgetTabImageVisualizer.listWidget_ImageList.currentItemChanged.connect(
            self.actionListWidget_ImageListChangeItem)

    # -------------------------- #
    # ----- Events Actions ----- #
    # -------------------------- #
    # ***** SET MAIN EVENTS ACTIONS *** #
    def actionButtonAdd(self):
        # Open dialog
        success, dialog = coFunc.openDirectoryDialog(
            classRef=self,
            dialogName='Open Table File',
            dialogOpenAt=self.str_pathToTheProject,
            dialogMultipleSelection=True)

        if success:  # if True
            dirName = dialog.split('/')[-1:][0]  # get the dirName
            # print(dialog)
            # print(dirName)
            # print()
            if dirName not in self.dict_tableDirsPaths.keys():  # if dir haven't added before
                self.addItemsToList(dialog)  # add file to the table list

            if self.listWidget_DirList.currentItem() is None:  # Set row 0 as current row
                self.listWidget_DirList.setCurrentRow(0)  # Set current row

            if self.dict_tableDirsPaths.keys().__len__() >= 1:
                self.buttonExecute.setEnabled(True)
            # self.prt_dict_tableFilePaths()

    def updateButtonRemove(self):
        self.updateClassesList()

    def actionButtonRemove(self):
        if self.listWidget_DirList.currentItem() is not None:  # if some item is selected
            self.dict_tableDirsPaths.pop(self.listWidget_DirList.currentItem().text(), None)  # Delete item from dict
            self.listWidget_DirList.takeItem(self.listWidget_DirList.currentRow())  # Delete item from widget

            # if there are not enough files loaded
            if self.dict_tableDirsPaths.keys().__len__() < 1:
                self.buttonExecute.setEnabled(False)  # disable the Execute Button

    # ***** SET CURRENT LIST ROW CHANGE *** #
    def actionDirListRowChanged_event(self):
        self.listWidget_ClassesList.clear()

        if self.listWidget_DirList.currentItem() is not None:
            currentDir = self.listWidget_DirList.currentItem().text()
            # print(currentDir)
            # print(currentClass)
            for _className_ in self.dict_tableDirsPaths[currentDir][self.dkeyClassesNames()]:
                self.listWidget_ClassesList.addItem(QListWidgetItem(_className_))
            self.listWidget_ClassesList.setCurrentRow(0)

    # ***** SET CURRENT LIST ROW CHANGE *** #
    def actionClassesListRowChanged_event(self):
        self.widgetTabImageVisualizer.clearImageList()

        if self.listWidget_DirList.currentItem() is not None:
            currentDir = self.listWidget_DirList.currentItem().text()
            currentClass = self.listWidget_ClassesList.currentItem().text()
            # print(currentDir)
            # print(currentClass)
            for _fileName_ in self.dict_tableDirsPaths[currentDir][self.dkeyClassesData()][currentClass][
                self.dkeyImageNames()]:
                self.widgetTabImageVisualizer.addItemToList(_fileName_)
            self.widgetTabImageVisualizer.setCurrentRow(0)

    # ***** IMAGE VISUALISER ***** #
    def actionListWidget_ImageListChangeItem(self):
        if self.listWidget_DirList.currentItem() is not None:
            currentDir = self.listWidget_DirList.currentItem().text()
            currentClass = self.listWidget_ClassesList.currentItem().text()
            currentImageName = self.widgetTabImageVisualizer.getCurrentRowItem()

            if currentImageName in self.dict_tableDirsPaths[currentDir][self.dkeyClassesData()][currentClass][
                self.dkeyImageNames()]:
                index = self.dict_tableDirsPaths[currentDir][self.dkeyClassesData()][currentClass][
                    self.dkeyImageNames()].index(currentImageName)
                currentImagePath = \
                    self.dict_tableDirsPaths[currentDir][self.dkeyClassesData()][currentClass][self.dkeyDataFullpath()][
                        index]
                # print(currentImagePath)
                self.widgetTabImageVisualizer.showImageInVisualiser(currentImagePath)
        else:
            self.widgetTabImageVisualizer.showImageInVisualiser(None)


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

    # -------------------------------- #
    # ----- ListWidget_ImageList ----- #
    # -------------------------------- #
    def clearImageList(self):
        self.listWidget_ImageList.clear()

    def addItemToList(self, str_itemName):
        self.listWidget_ImageList.addItem(QListWidgetItem(str_itemName))

    def setCurrentRow(self, row):
        self.listWidget_ImageList.setCurrentRow(row)

    def getCurrentRowItem(self):
        if self.listWidget_ImageList.currentItem() is not None:
            return self.listWidget_ImageList.currentItem().text()
        return None

    # ------------------------------------ #
    # ----- GLWindow_ImageVisualiser ----- #
    # ------------------------------------ #
    def showImageInVisualiser(self, imagePath=None):
        if imagePath is not None:
            # print(imagePath)
            img = cv2.imread(imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            # print(img.shape, img)
        else:
            img = None
        self.GLWindow_ImageVisualizer.setImage(img)


# *********** OpenGL Image Visualizer *********** #
class OpenGLWidgetImageVisualizer(QGLWidget):
    def __init__(self, parent=None):
        QGLWidget.__init__(self, parent)
        self._timer = QBasicTimer()  # creating timer
        self._timer.start(1000 / 60, self)  # setting up timer ticks to 60 fps

        self.setMinimumWidth(256)
        self.setMinimumHeight(256)
        self.imgToView = None

    def initializeGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.imgToView is not None:
            # print(self.imgToView.shape)
            glColor3f(0.0, 0.0, 0.0)
            w = self.imgToView.shape[1]
            h = self.imgToView.shape[0]
            original_ratio = w / h
            designer_ratio = self.width() / self.height()
            if original_ratio > designer_ratio:
                designer_height = self.width() / original_ratio
                scale = designer_height / h
            else:
                designer_width = self.height() * original_ratio
                scale = designer_width / w

            if self.width() < self.height():
                if original_ratio == 1.0:
                    glRasterPos2f(-1.0, -0.5)
                elif original_ratio > 1.0:
                    pos = scale * h
                    pos = 1.0 - ((self.height() - pos) / self.height())
                    glRasterPos2f(-1.0, -pos)
                else:
                    pos = scale * w
                    pos = 1.0 - ((self.width() - pos) / self.width())
                    glRasterPos2f(-1.0, -pos)
            elif self.width() > self.height():
                if original_ratio == 1.0:
                    glRasterPos2f(-0.5, -1.0)
                elif original_ratio > 1.0:
                    pos = scale * h
                    pos = 1.0 - ((self.height() - pos) / self.height())
                    glRasterPos2f(-pos, -1.0)
                else:
                    pos = scale * w
                    pos = 1.0 - ((self.width() - pos) / self.width())
                    glRasterPos2f(-pos, -1.0)
            else:
                glRasterPos2f(-1.0, -1.0)

            glPixelZoom(scale, scale)
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, self.imgToView)

        else:
            glColor3f(0.0, 0.0, 0.0)

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)
        glLoadIdentity()
        # Make the display area proportional to the size of the view
        glOrtho(-w / self.width(), w / self.width(), -h / self.height(), h / self.height(), -1.0, 1.0)

    def timerEvent(self, QTimerEvent):
        self.update()  # refreshing the widget

    def setImage(self, img):
        self.imgToView = img


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
