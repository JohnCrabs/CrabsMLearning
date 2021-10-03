import warnings

import lib.core.file_manipulation as file_manip
from PySide2.QtCore import (
    QUrl
)

from PySide2.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from PySide2.QtGui import (
    QIcon
)


def openFileDialog(classRef, dialogName='Pick a File', dialogOpenAt=file_manip.PATH_HOME, dialogFilters=None,
                   dialogMultipleSelection: bool = False):
    """
    A function to open a dialog for opening files.
    :param classRef: The class which will use the dialog
    :param dialogName: The dialog's name.
    :param dialogOpenAt: The path the dialog will be opened
    :param dialogFilters: The dialog's filter files
    :param dialogMultipleSelection: A boolean to tell to dialog if multiple selection is supported
    :return: True/False, dialog/None
    """
    if dialogFilters is None:  # if dialogFilter is None
        dialogFilters = ['All Files (*.*)']  # set default Value
    dialog = QFileDialog(classRef, dialogName)  # Open a Browse Dialog
    if dialogMultipleSelection:  # if True
        dialog.setFileMode(QFileDialog.ExistingFiles)  # Set multiple selection
    dialog.setDirectory(dialogOpenAt)  # Set default directory to the default project
    dialog.setSidebarUrls([QUrl.fromLocalFile(dialogOpenAt)])  # Open to default path
    dialog.setNameFilters(dialogFilters)  # Choose SPACE Files
    if dialog.exec_() == QFileDialog.Accepted:  # if path Accepted
        return True, dialog
    else:
        return False, None


def openDirectoryDialog(classRef, dialogName='Pick a Directory', dialogOpenAt=file_manip.PATH_HOME,
                        dialogMultipleSelection: bool = False):
    """
    A function to open a dialog for opening files.
    :param classRef: The class which will use the dialog
    :param dialogName: The dialog's name.
    :param dialogOpenAt: The path the dialog will be opened
    :param dialogMultipleSelection: A boolean to tell to dialog if multiple selection is supported
    :return: True/False, dialog/None
    """
    dialog = QFileDialog(classRef, dialogName)  # Open a Browse Dialog
    if dialogMultipleSelection:  # if True
        dialog.setFileMode(QFileDialog.ExistingFiles)  # Set multiple selection
    dialog.setDirectory(dialogOpenAt)  # Set default directory to the default project
    dialog.setSidebarUrls([QUrl.fromLocalFile(dialogOpenAt)])  # Open to default path
    dirPath = dialog.getExistingDirectory()
    if dirPath:  # if path Accepted
        return True, str(dirPath)
    else:
        return False, None


def consoleMessage(textMessageInfo):
    """
    Show a warning message on the console
    :param textMessageInfo: the message to be shown
    :return: Nothing
    """
    warnings.warn(textMessageInfo)  # show a warning message to console


def errorMessageDialog(classRef, errorType, textMessageInfo,
                       windowTitle="Hey, I have an Error Message for you!", iconPath=None):
    """
    A function for opening an error message dialog and printing a specified error type/message.
    :param classRef: The class the dialog refers to
    :param errorType: The type of the error (e.g. Error 404)
    :param textMessageInfo: The message of the error (e.g. File not found!)
    :param windowTitle: The title of the window
    :param iconPath: The icon of the dialog window
    :return: nothing
    """
    msg = QMessageBox(classRef)
    if iconPath is not None:
        msg.setWindowIcon(QIcon(iconPath))
    msg.setIcon(QMessageBox.Critical)  # Set the icon of the dialog window
    msg.setWindowTitle(windowTitle)  # Set the title of dialog window
    msg.setText(errorType)  # Set the errorType
    msg.setInformativeText(textMessageInfo)  # set the errorMessage
    msg.exec_()  # execute the dialog (show the message)
