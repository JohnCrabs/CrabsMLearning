import os
import csv
import warnings
import pandas as pd
import numpy as np
from shutil import copyfile
import datetime as dt

PATH_NORM_SLASH = os.path.normpath('/')
PATH_HOME = os.path.expanduser('~')
PATH_DOCUMENTS = os.path.expanduser('~/Documents')

SIZE_BYTE = 'byte'
SIZE_KB = 'kb'
SIZE_MB = 'mb'
SIZE_GB = 'gb'
SIZE_TB = 'tb'


def getCurrentDatetimeForPath():
    return dt.datetime.now().strftime("%d%m%Y_%H%M%S")


def getCurrentDatetimeForConsole():
    return dt.datetime.now().strftime("%d/%m/%Y_%H:%M:%S")


def checkAndCreateFolder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def checkAndCreateFolders(path):
    if not os.path.exists(path):
        os.makedirs(path)


def checkPathExistence(path):
    return os.path.exists(path)


def checkAndRenameExistPath_retPath(path):
    tmp_path = path
    index = 1
    while checkPathExistence(tmp_path):
        tmp_path = path + '_' + str(index)
        index += 1
    return tmp_path


def checkAndRenameExistPath_retName(dirPath, name):
    tmp_name = name
    tmp_path = dirPath + tmp_name
    index = 1
    while checkPathExistence(tmp_path):
        tmp_name = name + '_' + str(index)
        tmp_path = dirPath + tmp_name
        index += 1
    return tmp_name


def normPath(path):
    if os.path.isfile(path):
        return os.path.normpath(path)
    return os.path.normpath(path) + PATH_NORM_SLASH


def realPath(path):
    return os.path.realpath(path)


def getFileSize(path, size_type=SIZE_GB):
    file_size = os.stat(path).st_size

    if size_type is SIZE_KB:
        return file_size / 1024.0
    elif size_type is SIZE_MB:
        return file_size / (1024.0 * 1024.0)
    elif size_type is SIZE_GB:
        return file_size / (1024.0 * 1024.0 * 1024.0)
    elif size_type is SIZE_TB:
        return file_size / (1024.0 * 1024.0 * 1024.0 * 1024.0)
    else:
        return file_size


def copy_from_to(copy_from: str, copy_to_dir: str):
    norm_copy_from = normPath(copy_from)  # normalize the copy_form path
    checkAndCreateFolders(copy_to_dir)  # check if copy_to_dir path exists and create it if not
    norm_copy_to = normPath(copy_to_dir) + os.path.basename(norm_copy_from)  # normalise the copy_to path
    try:
        copyfile(norm_copy_from, norm_copy_to)  # copy the file to location
        return True, norm_copy_from, norm_copy_to  # return True and file path
    except IOError:
        return False, norm_copy_from, norm_copy_to  # return False and file path


def pathFileName(path):
    return os.path.basename(path)


def pathFileSuffix(path):
    return os.path.splitext(path)[1]


def getColumnNames(path):
    # find the file suffix (extension) and take is as lowercase without the comma
    suffix = os.path.splitext(path)[1].lower().split('.')[1]
    columns = []
    # Read the file Data - CSV
    if suffix == 'csv':
        # read only the needed columns
        fileData = pd.read_csv(path, nrows=1)
        columns = fileData.keys().tolist()
        # print(fileData.keys())
    elif suffix == 'xlsx':
        # read only the needed columns
        fileData = pd.read_excel(path, nrows=1)
        columns = fileData.keys().tolist()
    return columns


def exportCSV(csv_path: str, list_write: []):
    """
    A Function for writing CSV files.
    :param csv_path: The path for the csv to be exported.
    :param list_write: The list to be written in file
    :return: Nothing
    """
    if not os.path.exists(os.path.dirname(csv_path)):  # Check if path does not exist
        warnings.warn("Path does not exist! Path will be created!")  # Warn in console
        directory = os.path.dirname(csv_path)  # Find the directories in the path
        os.mkdir(directory)  # Create the directories

    with open(csv_path, 'w', newline='') as csvfile:  # Open the path to write the file
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in list_write:  # For each row
            csv_writer.writerow(row)  # Write row to file


def exportDictionaryNonList(dictForExport: {}, exportPath: str, headerLine=None):
    exportList = []
    if headerLine is not None:
        exportList.append(headerLine)
    for key in dictForExport.keys():
        tmp_row = [key, dictForExport[key]]
        exportList.append(tmp_row)
    exportCSV(exportPath, exportList)


def exportDictionaryList(dictForExport: {}, exportPath: str, headerLine=None):
    exportList = []
    if headerLine is not None:
        exportList.append(headerLine)
    for key in dictForExport.keys():
        tmp_list = dictForExport[key].copy()
        if type(tmp_list) is np.ndarray:
            tmp_list = tmp_list.tolist()
        for row in tmp_list:
            tmp_row = [key]
            if type(row) is list:
                tmp_row.extend(row)
            else:
                tmp_row.append(row)
            exportList.append(tmp_row)
    exportCSV(exportPath, exportList)

