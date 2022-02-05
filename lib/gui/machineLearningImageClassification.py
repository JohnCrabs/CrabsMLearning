import sys
import cv2 as cv2
import numpy as np
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
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QComboBox,
    QSpacerItem
)

from PySide2.QtGui import (
    # QIcon,
    QPixmap
)

from lib.core.project_flags import *
from lib.gui.commonFunctions import *
import lib.core.machineLearningClassification as mlc
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
        self._DKEY_CLASSES_ML_CODE: str = 'classes-ml-code'
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
        self.widgetTabMachineLearningSettings = WidgetTabMachineLearningSettings()  # create a tab for general settings

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

        self.mlc_Classification = mlc.MachineLearningImageClassification()
        self.mlc_Classification.setMLC_dict()

    # DICTIONARY FILE PARAMETERS
    def dkeyDirName(self):
        return self._DKEY_DIR_NAME

    def dkeyFullPath(self):
        return self._DKEY_FULLPATH

    def dkeyClassesNames(self):
        return self._DKEY_CLASSES_NAMES

    def dkeyClassesMLCode(self):
        return self._DKEY_CLASSES_ML_CODE

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
        self.widgetTabMachineLearningSettings.setWidget()  # Set the Tab Machine Learnings Settings
        self.mainTabWidget.addTab(self.widgetTabImageVisualizer, "Image Visualizer")  # Add it to mainTanWidget
        self.mainTabWidget.addTab(self.widgetTabMachineLearningSettings,
                                  "Machine Learning Settings")  # Add it to mainTabWidget

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
        """
        Create a dictionary as follows:
        <dir_name> -> {
                        "name": str,
                        "full-path": str,
                        "classes-names": [str],
                        "classes-names": [[int]]
                        "classes-data": {
                            <dir_name_class_0>: {},
                            <dir_name_class_1>: {},
                            ..................: {},
                            <dir_name_class_n>: {}}
                        }
        }

        <dir_name_class_0>: {
            "full-path": str,
            "image-names": [str],
            "data-size": int (the number of samples of the current class),
            data_fullpaths: [str] (this will be used later to create the x and y arrays)
        }

        :param fullPath:
        :return:
        """
        dirName = fullPath.split('/')[-1:][0]  # find the name of the file
        dirClasses = os.listdir(fullPath)
        # Create the dictionary
        self.dict_tableDirsPaths[dirName] = {
            self.dkeyDirName(): dirName,
            self.dkeyFullPath(): fullPath,
            self.dkeyClassesNames(): dirClasses,
            self.dkeyClassesMLCode(): nameClassList2MachineLearningList(dirClasses),
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
        self.buttonExecute.clicked.connect(self.actionButtonExecute)  # buttonExecute -> clicked

        # ListWidget Events
        self.listWidget_DirList.currentRowChanged.connect(self.actionDirListRowChanged_event)
        self.listWidget_ClassesList.currentRowChanged.connect(self.actionClassesListRowChanged_event)

    def setImageVisualiserEvents(self):
        self.widgetTabImageVisualizer.listWidget_ImageList.currentItemChanged.connect(
            self.actionListWidget_ImageListChangeItem)

    # -------------------------- #
    # ----- Events Actions ----- #
    # -------------------------- #
    # ***** SET MAIN EVENTS ACTIONS ***** #
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

    # ***** SET CURRENT DIR LIST ROW CHANGE ***** #
    def actionDirListRowChanged_event(self):
        self.listWidget_ClassesList.clear()

        if self.listWidget_DirList.currentItem() is not None:
            currentDir = self.listWidget_DirList.currentItem().text()
            # print(currentDir)
            # print(currentClass)
            for _className_ in self.dict_tableDirsPaths[currentDir][self.dkeyClassesNames()]:
                self.listWidget_ClassesList.addItem(QListWidgetItem(_className_))
            self.listWidget_ClassesList.setCurrentRow(0)

    # ***** SET CURRENT CLASSES LIST ROW CHANGE ***** #
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

    # *********************************************************** #
    # *********** Helping Functions for ButtonExecute *********** #
    # *********************************************************** #
    # *                                                         * #
    def BE_Create_XY_dict(self, key):
        XY_dict = {'x': [], 'y': []}
        for _className_ in self.dict_tableDirsPaths[key][self.dkeyClassesData()].keys():
            # Get the index of the current class
            mlCodeIndex = self.dict_tableDirsPaths[key][self.dkeyClassesNames()].index(_className_)
            # Get the code of the current class using the above index
            mlCode = self.dict_tableDirsPaths[key][self.dkeyClassesMLCode()][mlCodeIndex]
            for _path_ in self.dict_tableDirsPaths[key][self.dkeyClassesData()][_className_][self.dkeyDataFullpath()]:
                XY_dict['x'].append(_path_)
                XY_dict['y'].append(mlCode)

        return XY_dict

    # *                                                         * #
    # *********************************************************** #

    # ***** EXECUTION ***** #
    def actionButtonExecute(self):
        # If true run the main pipeline
        if self.dict_tableDirsPaths.keys().__len__() > 0:  # if there is at least a file (safety if)
            # 00 - Error Checking
            # 01 - Run The Main Routine
            XY_dict = {}
            # for each folder in directory
            for _folderName_ in self.dict_tableDirsPaths.keys():
                # Set a path for exporting the input-output data
                currentFileName = os.path.splitext(_folderName_)[0]  # get Folder Name
                currentDatetime = file_manip.getCurrentDatetimeForPath()  # Find Current Datetime
                exportPrimaryDir = 'export_folder' + \
                                   '/' + currentFileName + '/' + currentDatetime
                # exportDataFolder = os.path.normpath(exportPrimaryDir + '/Data')

                XY_dict[_folderName_] = self.BE_Create_XY_dict(
                    _folderName_)  # Find the classes in dir and set the dictionary (x,y)
                fullSizeOfData = XY_dict[_folderName_]['x'].__len__()  # Get the size of the dataset

                indexList = np.random.permutation(fullSizeOfData).tolist()  # Calculate a list of indexes (permutation)
                sliceIndex = fullSizeOfData - int(fullSizeOfData * 0.25)  # Find the index for slicing
                trainIndexes = np.array(indexList[:sliceIndex])  # Set train-validation indexes
                testIndexes = np.array(indexList[sliceIndex:])  # Set test indexes

                # Create the Train-Validation set
                X_train_paths = np.array(XY_dict[_folderName_]['x'])[trainIndexes]
                X_train_images = []
                Y_train = np.array(XY_dict[_folderName_]['y'])[trainIndexes]

                # Create the Test set
                X_test_paths = np.array(XY_dict[_folderName_]['x'])[testIndexes]
                X_test_images = []
                Y_test = np.array(XY_dict[_folderName_]['y'])[testIndexes]

                # Read images for train set
                for _path_ in X_train_paths:
                    img = cv2.imread(_path_)
                    # print(img.shape)
                    X_train_images.append(img)
                X_train_images = np.array(X_train_images)

                # Read images for test set
                for _path_ in X_test_paths:
                    img = cv2.imread(_path_)
                    # print(img.shape)
                    X_test_images.append(img)
                X_test_images = np.array(X_test_images)

                # Execute the Classification
                self.mlc_Classification.fit(X_TrainVal=X_train_images,
                                            y_TrainVal=Y_train,
                                            X_Test=X_test_images,
                                            y_Test=Y_test,
                                            exportFolder=exportPrimaryDir)


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

            pos_w = scale * w
            pos_w = 1.0 - ((self.width() - pos_w) / self.width())
            glRasterPos2f(-pos_w, -1.0)
            pos_h = scale * h
            pos_h = 1.0 - ((self.height() - pos_h) / self.height())
            glRasterPos2f(-pos_w, -pos_h)

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


# *********** Machine Learning Settings *********** #
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

    def setWidget(self):
        """
            A function to create the widget components into the main QWidget
            :return: Nothing
        """
        self.tabGeneral.setWidget()  # set tab General (info)
        self.mainTabWidget.addTab(self.tabGeneral, "General")  # add tab to mainTabWidget
        self.vbox_main_layout.addWidget(self.mainTabWidget)  # add tabWidget to main layout


# *********** Machine Learning Settings --> General *********** #
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

        # create a spinBox for the MachineLearningMethodIndex
        self.spinBox_MachineLearningMethodsIndex = QSpinBox()
        # set the minimum value to 1 (at least a row needs to be used as input/output)
        self.spinBox_MachineLearningMethodsIndex.setMinimum(1)

        # -------------------------- #
        # ----- QDoubleSpinBox ----- #
        # -------------------------- #
        # create a doubleSpinBox for the test percentage
        self.doubleSpinBox_TestPercentage = QDoubleSpinBox()
        # set minimum value
        self.doubleSpinBox_TestPercentage.setMinimum(MLF_TEST_PERCENTAGE_MIN)
        # set maximum value
        self.doubleSpinBox_TestPercentage.setMaximum(MLF_TEST_PERCENTAGE_MAX)
        # set step
        self.doubleSpinBox_TestPercentage.setSingleStep(MLF_TEST_STEP_WIDGET)
        # create a doubleSpinBox fot the holdout percentage
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

        # --------------------- #
        # ----- QComboBox ----- #
        # --------------------- #
        # MachineLearningMethods
        self.comboBox_MachineLearningMethods = QComboBox()
        self.comboBox_MachineLearningMethods.setMinimumWidth(250)
        self.comboBox_MachineLearningMethods.addItems(MLPF_METHOD_LIST_IMAGE_CLASSIFICATION)

        # MultifileTrainingProcessing
        self.comboBox_MultifileTrainingProcessing = QComboBox()
        self.comboBox_MultifileTrainingProcessing.setMinimumWidth(200)
        self.comboBox_MultifileTrainingProcessing.addItems(MLPF_MULTIFILE_TRAINING_PROCESSING_LIST)

        # TestPercentageDistribution
        self.comboBox_TestPercentageDistribution = QComboBox()
        self.comboBox_TestPercentageDistribution.setMinimumWidth(200)
        self.comboBox_TestPercentageDistribution.addItems(MLPF_PERCENTAGE_DISTRIBUTION_LIST)

        # HoldoutPercentageDistribution
        self.comboBox_HoldoutPercentageDistribution = QComboBox()
        self.comboBox_HoldoutPercentageDistribution.setMinimumWidth(200)
        self.comboBox_HoldoutPercentageDistribution.addItems(MLPF_PERCENTAGE_DISTRIBUTION_LIST)

        # ------------------------------ #
        # ----- Set Default Values ----- #
        # ------------------------------ #
        self._experimentNumberDefaultValue = MLF_DEFAULT_EXPERIMENT_VALUE
        self._testPercentageDefaultValue = MLF_DEFAULT_TEST_PERCENTAGE
        self._holdoutPercentageDefaultValue = MLF_DEFAULT_HOLDOUT_PERCENTAGE
        self._exportPathDefaultValue = MLF_DEFAULT_EXPORT_FOLDER_PATH
        self._mlMethodDefaultValue = MLF_DEFAULT_METHOD
        self._mlMethodIndexDefaultValue = MLF_DEFAULT_METHOD_INDEX
        self._mlMultifileTrainingProcessingDefaultValue = MLF_DEFAULT_MULTIFILE_TRAINING_PROCESSING
        self._testPercentageDistributionDefaultValue = MLF_DEFAULT_TEST_PERCENTAGE_DISTRIBUTION
        self._holdoutPercentageDistributionDefaultValue = MLF_DEFAULT_HOLDOUT_PERCENTAGE_DISTRIBUTION

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
        # label_ExperimentalNumber.setMinimumWidth(100)
        label_TestPercentage = QLabel("Test Percentage (0.00-0.75):")
        # label_TestPercentage.setMinimumWidth(100)
        label_TestPercentageDistribution = QLabel("Test Percentage Distribution:")
        # label_TestPercentageDistribution.setMinimumWidth(100)
        label_HoldOutPercentage = QLabel("Hold-Out Percentage (0.00-0.75):")
        # label_HoldOutPercentage.setMinimumWidth(100)
        label_HoldoutPercentageDistribution = QLabel("Hold-Out Percentage Distribution:")
        # label_HoldoutPercentageDistribution.setMinimumWidth(100)
        label_SetOutPath = QLabel("Export Folder:")
        # label_SetOutPath.setMinimumWidth(100)
        label_MachineLearningMethod = QLabel("Machine Learning Method:")
        # label_MachineLearningMethod.setMinimumWidth(100)
        label_MachineLearningMethodIndex = QLabel("Method Usage Index:")
        # label_MachineLearningMethodIndex.setMinimumWidth(100)
        label_MultifileTrainingProcessing = QLabel("Multifile Training Processing:")
        # label_MultifileTrainingProcessing.setMinimumWidth(100)

        # Set SpinBoxes
        hbox_ExperimentalNumber = QHBoxLayout()
        hbox_ExperimentalNumber.addWidget(label_ExperimentalNumber)
        hbox_ExperimentalNumber.addWidget(self.spinBox_ExperimentNumber)
        hbox_ExperimentalNumber.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TestPercentage = QHBoxLayout()
        hbox_TestPercentage.addWidget(label_TestPercentage)
        hbox_TestPercentage.addWidget(self.doubleSpinBox_TestPercentage)
        hbox_TestPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_TestPercentageDistribution = QHBoxLayout()
        hbox_TestPercentageDistribution.addWidget(label_TestPercentageDistribution)
        hbox_TestPercentageDistribution.addWidget(self.comboBox_TestPercentageDistribution)
        hbox_TestPercentageDistribution.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_HoldOutPercentage = QHBoxLayout()
        hbox_HoldOutPercentage.addWidget(label_HoldOutPercentage)
        hbox_HoldOutPercentage.addWidget(self.doubleSpinBox_HoldoutPercentage)
        hbox_HoldOutPercentage.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_HoldOutPercentageDistribution = QHBoxLayout()
        hbox_HoldOutPercentageDistribution.addWidget(label_HoldoutPercentageDistribution)
        hbox_HoldOutPercentageDistribution.addWidget(self.comboBox_HoldoutPercentageDistribution)
        hbox_HoldOutPercentageDistribution.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_finalSpinBoxes = QVBoxLayout()
        vbox_finalSpinBoxes.addLayout(hbox_ExperimentalNumber)
        vbox_finalSpinBoxes.addLayout(hbox_TestPercentage)
        vbox_finalSpinBoxes.addLayout(hbox_TestPercentageDistribution)
        vbox_finalSpinBoxes.addLayout(hbox_HoldOutPercentage)
        vbox_finalSpinBoxes.addLayout(hbox_HoldOutPercentageDistribution)

        # SetOutPath Layout
        hbox_SetOutPath = QHBoxLayout()
        hbox_SetOutPath.addWidget(label_SetOutPath)
        hbox_SetOutPath.addWidget(self.lineEdit_SetOutPath)
        hbox_SetOutPath.addWidget(self.buttonSetOutPath)
        # hbox_SetOutPath.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        # Button Layout
        hbox_buttons = QHBoxLayout()
        hbox_buttons.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))
        hbox_buttons.addWidget(self.buttonRestoreDefault)

        # MachineLearningMethods
        hbox_MachineLearningMethods = QHBoxLayout()
        hbox_MachineLearningMethods.addWidget(label_MachineLearningMethod)
        hbox_MachineLearningMethods.addWidget(self.comboBox_MachineLearningMethods)
        hbox_MachineLearningMethods.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        hbox_MachineLearningMethodsIndex = QHBoxLayout()
        hbox_MachineLearningMethodsIndex.addWidget(label_MachineLearningMethodIndex)
        hbox_MachineLearningMethodsIndex.addWidget(self.spinBox_MachineLearningMethodsIndex)
        hbox_MachineLearningMethodsIndex.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        vbox_finalMachineLearningMethods = QVBoxLayout()
        vbox_finalMachineLearningMethods.addLayout(hbox_MachineLearningMethods)
        vbox_finalMachineLearningMethods.addLayout(hbox_MachineLearningMethodsIndex)

        # MultifileTrainingProcessing
        hbox_MultifileTrainingProcessing = QHBoxLayout()
        hbox_MultifileTrainingProcessing.addWidget(label_MultifileTrainingProcessing)
        hbox_MultifileTrainingProcessing.addWidget(self.comboBox_MultifileTrainingProcessing)
        hbox_MultifileTrainingProcessing.addSpacerItem(QSpacerItem(INT_MAX_STRETCH, 0))

        self.vbox_main_layout.addLayout(vbox_finalSpinBoxes)
        self.vbox_main_layout.addLayout(hbox_SetOutPath)
        self.vbox_main_layout.addLayout(vbox_finalMachineLearningMethods)
        self.vbox_main_layout.addLayout(hbox_MultifileTrainingProcessing)
        # self.vbox_main_layout.addSpacerItem(QSpacerItem(0, INT_MAX_STRETCH))
        self.vbox_main_layout.addLayout(hbox_buttons)

    def restoreDefaultValues(self):
        # set default value
        self.spinBox_ExperimentNumber.setValue(self._experimentNumberDefaultValue)
        self.doubleSpinBox_TestPercentage.setValue(self._testPercentageDefaultValue)
        self.doubleSpinBox_HoldoutPercentage.setValue(self._holdoutPercentageDefaultValue)
        self.lineEdit_SetOutPath.setText(os.path.normpath(self._exportPathDefaultValue))
        self.comboBox_MachineLearningMethods.setCurrentText(self._mlMethodDefaultValue)
        self.spinBox_MachineLearningMethodsIndex.setValue(self._mlMethodIndexDefaultValue)
        self.comboBox_MultifileTrainingProcessing.setCurrentText(self._mlMultifileTrainingProcessingDefaultValue)
        self.comboBox_TestPercentageDistribution.setCurrentText(self._testPercentageDistributionDefaultValue)
        self.comboBox_HoldoutPercentageDistribution.setCurrentText(self._holdoutPercentageDistributionDefaultValue)

    def setEvents_(self):
        self.buttonRestoreDefault.clicked.connect(self.restoreDefaultValues)

    # ------------------------------ #
    # ----- GET DEFAULT VALUES ----- #
    # ------------------------------ #
    def getDefaultExperimentNumber(self):
        return self._experimentNumberDefaultValue

    def getDefaultTestPercentage(self):
        return self._testPercentageDefaultValue

    def getDefaultTestPercentageDistribution(self):
        return self._testPercentageDistributionDefaultValue

    def getDefaultHoldoutPercentage(self):
        return self._holdoutPercentageDefaultValue

    def getDefaultHoldoutPercentageDistribution(self):
        return self._holdoutPercentageDistributionDefaultValue

    def getDefaultExportPath(self):
        return self._exportPathDefaultValue

    def getDefaultMethod(self):
        return self._mlMethodDefaultValue

    def getDefaultMethodIndex(self):
        return self._mlMethodIndexDefaultValue

    def getDefaultMultifileTrainingProcessing(self):
        return self._mlMultifileTrainingProcessingDefaultValue


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
