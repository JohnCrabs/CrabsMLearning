import os
import pandas as pd
from shutil import copyfile

PATH_NORM_SLASH = os.path.normpath('/')
PATH_HOME = os.path.expanduser('~')
PATH_DOCUMENTS = os.path.expanduser('~/Documents')

SIZE_BYTE = 'byte'
SIZE_KB = 'kb'
SIZE_MB = 'mb'
SIZE_GB = 'gb'
SIZE_TB = 'tb'


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
