import os
import lib.core.file_manipulation as file_manip
import tkinter as tk

NEW_PROJECT_DEFAULT_FOLDER = file_manip.PATH_HOME
DOCUMENTS_FOLDER = file_manip.PATH_DOCUMENTS
DOC_PROJECT_FOLDER = DOCUMENTS_FOLDER + '/CrabsMLearning'
PROJECT_FOLDER = os.path.normpath(os.path.realpath(__file__) + '/../../../')

INT_SCREEN_WIDTH = tk.Tk().winfo_screenwidth()  # get the screen width
INT_SCREEN_HEIGHT = tk.Tk().winfo_screenheight()  # get the screen height
INT_WIN_WIDTH = 1024  # this variable is only for the if __name__ == "__main__"
INT_WIN_HEIGHT = 512  # this variable is only for the if __name__ == "__main__"

INT_MAX_STRETCH = 100000  # Spacer Max Stretch
INT_BUTTON_MIN_WIDTH = 50  # Minimum Button Width
INT_BUTTON_MIN_HEIGHT = 50  # Minimum Button Width
INT_ADD_REMOVE_BUTTON_SIZE = 48

ICON_ADD = PROJECT_FOLDER + "/icon/add_cross_128x128.png"
ICON_REMOVE = PROJECT_FOLDER + "/icon/remove_line_128x128.png"
ICON_ADD_FILLED = PROJECT_FOLDER + "/icon/add_cross_128x128_filled.png"
ICON_REMOVE_FILLED = PROJECT_FOLDER + "/icon/remove_line_128x128_filled.png"

# --- PLOT INFO --- #
PLOT_FONTSIZE_TITLE = 25
PLOT_FONTSIZE_TICKS = 20
PLOT_FONTSIZE_LABEL = 22.5
PLOT_FONTSIZE_LEGEND = 18
PLOT_SIZE_WIDTH = 12.40
PLOT_SIZE_HEIGHT = 12.40
PLOT_SIZE_DPI = 100

TRAIN_TEST_SEPARATOR = 'Train/Test Split'
REAL_STYLE = ['bs-']
PRED_STYLE = ['go-']
