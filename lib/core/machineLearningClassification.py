import os
import datetime as dt

import joblib
import numpy as np
import openpyxl as op

import lib.core.file_manipulation as file_manip

from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

from sklearn.model_selection import (
    GridSearchCV,
    # RandomizedSearchCV,
    # cross_val_score
)

import tensorflow.keras as keras
import keras_tuner as kt

PATH_DEFAULT_EXPORT_DATA = os.path.normpath(file_manip.PATH_DOCUMENTS + '/MachineLearningRegression')
DEBUG_MESSAGES = True
H5_SUFFIX = '.h5'

###########################################
# ***** MACHINE LEARNING TECHNIQUES ***** #
#                                         #

MLR_CLF_LSTM_ROCK_PAPER_SCISSOR = 'LongShortTermMemoryNetwork_for_RockPaperScissor'

MLR_CLF_METHODS = [
    MLR_CLF_LSTM_ROCK_PAPER_SCISSOR
]

_MLR_NO_TUNING_LIST = [
]

_MLR_TUNING_NON_DEEP_METHODS = [
]

_MLR_TUNING_DEEP_METHODS = [
]

_MLR_3RD_DIM_DEEP_METHODS = [
]

#                                         #
###########################################

############################################
# ****** MACHINE LEARNING VARIABLES ****** #
#                                          #

MLR_KEY_METHOD = 'Method'
MLR_KEY_CUSTOM_PARAM = 'Custom Parameters'
MLR_KEY_PARAM_GRID = 'Grid Parameter'
MLR_KEY_TRAINED_MODEL = 'Trained Model'
MLR_KEY_3RD_DIM_SIZE = '3rd Dimension Size'

MLR_KEY_ALPHA = 'alpha'
MLR_KEY_TOL = 'tol'
MLR_KEY_SOLVER = 'solver'
MLR_KEY_KERNEL = 'kernel'
MLR_KEY_DEGREE = 'degree'
MLR_KEY_GAMMA = 'gamma'

MLR_KEY_ACTIVATION_FUNCTION = 'activation_function'
MLR_KEY_NUMBER_OF_EPOCHS = 'epochs'

MLR_SOLVER_AUTO = 'auto'
MLR_SOLVER_SVD = 'svd'
MLR_SOLVER_CHOLESKY = 'cholesky'
MLR_SOLVER_LSQR = 'lsqr'
MLR_SOLVER_SPARSE_CG = 'sparse_cg'
MLR_SOLVER_SAG = 'sag'
MLR_SOLVER_SAGA = 'saga'
MLR_SOLVER_LBFGS = 'lbfgs'

MLR_SOLVER_OPTIONS = [
    MLR_SOLVER_AUTO,
    MLR_SOLVER_SVD,
    MLR_SOLVER_CHOLESKY,
    # MLR_SOLVER_LSQR,
    MLR_SOLVER_SPARSE_CG,
    # MLR_SOLVER_SAG,
    # MLR_SOLVER_SAGA,
    # MLR_SOLVER_LBFGS
]

MLR_TOL_LIST = [
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10
]

MLR_EXEC_STATE = False

MLR_KERNEL_LINEAR = 'linear'
MLR_KERNEL_POLY = 'poly'
MLR_KERNEL_RBF = 'rbf'
MLR_KERNEL_SIGMOID = 'sigmoid'
MLR_KERNEL_PRECOMPUTED = 'precomputed'

MLR_KERNEL_OPTIONS = [
    'linear',
    'poly',
    'rbf',
    'sigmoid',
    'precomputed'
]

MLR_GAMMA_SCALE = 'scale'
MLR_GAMMA_AUTO = 'auto'

MLR_GAMMA_OPTIONS = [
    'scale',
    'auto'
]

DMLR_ACTIVATION_FUNCTIONS = [
    'relu',
    'sigmoid',
    'softmax',
    'softplus',
    'softsign',
    'tanh',
    'selu',
    'elu',
    'linear',
    # 'exponential'
]
DMLR_EPOCHS = 500


#                                          #
############################################

# A class to store the Machine Learning Regression algorithms
class MachineLearningRegression:
    def __init__(self):
        self._MLR_dictMethods = {}

        self._MLR_KEY_MIN = 'min'
        self._MLR_KEY_MAX = 'max'
        self._MLR_KEY_STEP = 'step'
        self._MLR_KEY_STATE = 'state'

        #########################################
        # ***** MACHINE LEARNING DEFAULTS ***** #
        #                                       #

        # ----- RIDGE ----- #
        self._MLR_RIDGE_ALPHA_DEFAULT_VALUE = 1.0
        self._MLR_RIDGE_ALPHA_DEFAULT = [1.0]
        self._MLR_RIDGE_TOL_DEFAULT = [1e-3]
        self._MLR_RIDGE_SOLVER_DEFAULT = [MLR_SOLVER_AUTO]

        # ----- SVR ----- #
        self._MLR_SVR_KERNEL_DEFAULT = ['rbf']
        self._MLR_SVR_DEGREE_DEFAULT = [3]
        self._MLR_SVR_GAMMA_DEFAULT = ['scale']
        self._MLR_SVR_TOL_DEFAULT = [1e-3]

        #                                       #
        #########################################

    def setMLR_dict(self):
        for _method_ in MLR_CLF_METHODS:
            self._MLR_dictMethods[_method_] = {}

            self.restore_LSTM_RockPaperScissor_Defaults()

    # ********************************** #
    # ***** RESTORE DEFAULT VALUES ***** #
    # ********************************** #
    def restore_LSTM_RockPaperScissor_Defaults(self):
        self._MLR_dictMethods[MLR_CLF_LSTM_ROCK_PAPER_SCISSOR] = {
            MLR_KEY_METHOD: self.DeepLearning_RockPaperScissor_LSTM,
            self._MLR_KEY_STATE: MLR_EXEC_STATE,
            MLR_KEY_PARAM_GRID: {
                MLR_KEY_ACTIVATION_FUNCTION: DMLR_ACTIVATION_FUNCTIONS,
                MLR_KEY_NUMBER_OF_EPOCHS: DMLR_EPOCHS
            },
            MLR_KEY_TRAINED_MODEL: None
        }

    # ********************************* #
    # ***** DEEP LEARNING METHODS ***** #
    # ********************************* #
    @staticmethod
    def DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model, directory, name,
                         epochs=100, tuner_objective='loss'):
        tuner = kt.Hyperband(ffunc_build_model,
                             objective=tuner_objective,
                             max_epochs=50,
                             factor=5,
                             directory=directory,
                             project_name=name)

        # tuner = kt.RandomSearch(ffunc_build_model,
        #                         objective='accuracy',
        #                         max_trials=30,
        #                         directory=directory,
        #                         project_name=name
        #                         )

        tuner.search(train_x, train_y, epochs=epochs, validation_split=0.2)

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_x, train_y, epochs=epochs, validation_split=0.2)
        val_acc_per_epoch = history.history['val_loss']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

        model = tuner.hypermodel.build(best_hps)

        # Retrain the model
        model.fit(train_x, train_y, epochs=best_epoch, validation_split=0.2)

        eval_result = model.evaluate(test_x, test_y)
        print("test loss:", round(eval_result, 5))

        return model

    def DeepLearning_RockPaperScissor_LSTM(self, train_x, train_y, test_x, test_y,
                                           epochs: int, exportDirectory: str, activation_function_list: []):
        model = keras.models.Sequential([
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ])

    # ************************ #
    # ***** MAIN EXECUTE ***** #
    # ************************ #
    def fit(self, X_TrainVal: np.ndarray, y_TrainVal: np.ndarray,
            X_Test: np.ndarray, y_Test: np.ndarray, exportFolder=PATH_DEFAULT_EXPORT_DATA,
            validationPercentage: float = 0.25):
        # Set variables
        currentDatetime = dt.datetime.now().strftime("%d%m%Y_%H%M%S")  # Find Current Datetime
        errorFileName = 'PerformanceScores.xlsx'  # The file name to store the Performance Scores
        listStr_ModelPaths = []  # List to store the paths of the models (return it later)
        # The path to to the folder which we will export the models
        exportBaseDir = os.path.normpath(exportFolder + '/' + currentDatetime)
        exportTrainedModelsPath = os.path.normpath(exportFolder + '/' +
                                                   currentDatetime + '/TrainedModels') + '/'
        exportDeepLearningTunersPath = os.path.normpath(exportFolder + '/' +
                                                        currentDatetime + '/TrainedModels/DeepLearningTuners') + '/'
        # The path the folder which we will export the performance scores
        workbookDirPath = os.path.normpath(exportFolder + '/' +
                                           currentDatetime) + '/'

        workbookFilePath = workbookDirPath + errorFileName

        inputData_TrainVal = X_TrainVal  # store X_TrainVal to a new variable
        outputData_TrainVal = y_TrainVal  # store y_TrainVal to a new variable
        inputData_Test = X_Test  # store X_Test to a new variable
        outputData_Test = y_Test  # store y_Test to a new variable

        inputData_TrainVal_Shape = inputData_TrainVal.shape  # take the shape of inputTrainVal
        outputData_TrainVal_Shape = outputData_TrainVal.shape  # take the shape of outputTrainVal
        inputData_Test_Shape = inputData_Test.shape  # take the shape of inputTest
        outputData_Test_Shape = outputData_Test.shape  # take the shape of outputTest

        # Check if the exportFolder exists and if not create it
        file_manip.checkAndCreateFolders(os.path.normpath(exportFolder))  # exportFolder
        file_manip.checkAndCreateFolders(workbookDirPath)  # workbookDirPath
        file_manip.checkAndCreateFolders(exportTrainedModelsPath)  # exportTrainedModelsPath
        file_manip.checkAndCreateFolders(exportDeepLearningTunersPath)  # exportDeepLearningTunersPath

        # Create the Train and Validation Indexes
        randomIndexes = np.random.permutation(inputData_TrainVal_Shape[0])  # random permutation
        # Train Indexes
        trainIdxs = np.sort(randomIndexes[int(np.round(validationPercentage * len(randomIndexes))) + 1:])
        # Test Indexes
        valIdxs = np.sort(randomIndexes[:int(np.round(validationPercentage * len(randomIndexes)))])

        # ****************************************************************** #
        # ************************* DEBUG MESSAGES ************************* #
        if DEBUG_MESSAGES:
            print("CurrentDatetime = ", currentDatetime)
            print("ErrorFileName = ", errorFileName)
            print("ExportTrainedModelPaths = ", exportTrainedModelsPath)
            print("ExportTrainedModelPaths = ", exportDeepLearningTunersPath)
            print("WorkbookFilePath = ", workbookDirPath)
            print("InputData_TrainVal_Shape = ", inputData_TrainVal_Shape)
            print("OutputData_TrainVal_Shape = ", outputData_TrainVal_Shape)
            print("InputData_Test_Shape = ", inputData_Test_Shape)
            print("OutputData_Test_Shape = ", outputData_Test_Shape)
            print("Random_Indexes_Length = ", randomIndexes.__len__())
            print("Train_Indexes_Length = ", trainIdxs.__len__())
            print("Validation_Indexes_Length = ", valIdxs.__len__())
        # ****************************************************************** #

        for _methodKey_ in self._MLR_dictMethods.keys():  # for each method
            model = None  # a parameter to store the model
            modelName = _methodKey_  # store the methodKey to modelName
            realTrain = None  # a parameter to store the real expanded y_Train values
            realTest = None  # a parameter to store the real expanded y_Test values
            predTrain = None  # a parameter to store the predicted y_Train values
            predTest = None  # a parameter to store the predicted y_Test values

            if _methodKey_ in _MLR_NO_TUNING_LIST:  # if method cannot be tuning (e.g. LinearRegression)
                if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                    print(
                        file_manip.getCurrentDatetimeForConsole() + "::Training " + _methodKey_ + "...")  # console message
                    model = self._MLR_dictMethods[_methodKey_][MLR_KEY_METHOD]
                    model.fit(inputData_TrainVal,
                              outputData_TrainVal)  # model.fit()
                    print(file_manip.getCurrentDatetimeForConsole() + "::...COMPLETED!")  # console message
                    # Export model
                    modelExportPath = os.path.normpath(exportTrainedModelsPath + modelName + '_' +
                                                       currentDatetime + H5_SUFFIX)
                    listStr_ModelPaths.append(modelExportPath)
                    joblib.dump(model, modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Model exported at: ", modelExportPath)
                    self._MLR_dictMethods[_methodKey_][MLR_KEY_TRAINED_MODEL] = model

            elif _methodKey_ in _MLR_TUNING_NON_DEEP_METHODS:  # elif method is not a tf.keras
                if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                    print(
                        file_manip.getCurrentDatetimeForConsole() + "::Training " + _methodKey_ + "..")  # console message
                    # run Grid Search CV
                    model = GridSearchCV(self._MLR_dictMethods[_methodKey_][MLR_KEY_METHOD],
                                         self._MLR_dictMethods[_methodKey_][MLR_KEY_PARAM_GRID],
                                         n_jobs=-1)
                    model.fit(inputData_TrainVal,
                              outputData_TrainVal)  # model.fit()
                    print(file_manip.getCurrentDatetimeForConsole() + "::...COMPLETED!")  # console message
                    self._MLR_dictMethods[_methodKey_][MLR_KEY_TRAINED_MODEL] = model.best_estimator_

                    # Export model
                    modelExportPath = os.path.normpath(exportTrainedModelsPath + modelName + '_' +
                                                       currentDatetime + H5_SUFFIX)
                    listStr_ModelPaths.append(modelExportPath)
                    joblib.dump(model, modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Model exported at: ", modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Best Estimator = ", model.best_estimator_)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Best Score = ", model.best_score_)

            elif _methodKey_ in _MLR_TUNING_DEEP_METHODS:  # elif method is keras
                if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                    print(
                        file_manip.getCurrentDatetimeForConsole() + "::Training " + _methodKey_ + "..")  # console message

                    epochs = self._MLR_dictMethods[_methodKey_][MLR_KEY_PARAM_GRID][MLR_KEY_NUMBER_OF_EPOCHS]
                    # epochs = 500
                    activationFunctionList = self._MLR_dictMethods[_methodKey_][MLR_KEY_PARAM_GRID][
                        MLR_KEY_ACTIVATION_FUNCTION]

                    model = self._MLR_dictMethods[_methodKey_][MLR_KEY_METHOD](inputData_TrainVal,
                                                                               outputData_TrainVal,
                                                                               inputData_Test,
                                                                               outputData_Test,
                                                                               epochs,
                                                                               exportDeepLearningTunersPath,
                                                                               activationFunctionList
                                                                               )
                    print(file_manip.getCurrentDatetimeForConsole() + "::...COMPLETED!")  # console message
                    # Export model
                    modelExportPath = os.path.normpath(exportTrainedModelsPath + modelName + '_' +
                                                       currentDatetime + H5_SUFFIX)
                    listStr_ModelPaths.append(modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Model exported at: ", modelExportPath)
                    self._MLR_dictMethods[_methodKey_][MLR_KEY_TRAINED_MODEL] = model
                    self._MLR_dictMethods[_methodKey_][MLR_KEY_TRAINED_MODEL].save(modelExportPath)

            else:
                pass

            if model is not None:
                # if model is scikit learn model
                if model not in _MLR_NO_TUNING_LIST:
                    predTrain = model.predict(inputData_TrainVal)  # make predictions (train)
                    predTest = model.predict(inputData_Test)  # make predictions (test)
                    realTrain = outputData_TrainVal.copy()
                    realTest = outputData_Test.copy()
                    realTrain = np.array(realTrain).T
                    realTest = np.array(realTest).T
                    predTrain = np.array(predTrain).T
                    predTest = np.array(predTest).T

                    # print(realTest.shape)
                    # print(realTrain.shape)
                    # print(predTest.shape)
                    # print(predTrain.shape)

                else:
                    pass

                # pre-allocating results array for raw predicted values
                # PerObservationPoint
                errorMAE_Train = np.zeros(outputData_TrainVal_Shape[1])
                errorMSE_Train = np.zeros(outputData_TrainVal_Shape[1])
                errorMaxError_Train = np.zeros(outputData_TrainVal_Shape[1])
                errorMAE_Test = np.zeros(outputData_Test_Shape[1])
                errorMSE_Test = np.zeros(outputData_Test_Shape[1])
                errorMaxError_Test = np.zeros(outputData_Test_Shape[1])

                # print(outputData_TrainVal_Shape[1])
                for _index_ in range(0, outputData_TrainVal_Shape[1]):
                    # normalized first
                    errorMAE_Train[_index_] = \
                        mean_absolute_error(realTrain[_index_],
                                            predTrain[_index_])
                    errorMSE_Train[_index_] = \
                        mean_squared_error(realTrain[_index_],
                                           predTrain[_index_])
                    errorMaxError_Train[_index_] = \
                        max_error(realTrain[_index_],
                                  predTrain[_index_])
                    errorMAE_Test[_index_] = \
                        mean_absolute_error(realTest[_index_],
                                            predTest[_index_])
                    errorMSE_Test[_index_] = \
                        mean_squared_error(realTest[_index_],
                                           predTest[_index_])
                    errorMaxError_Test[_index_] = \
                        max_error(realTest[_index_],
                                  predTest[_index_])

                # Create the list with errors
                new_row = [modelName]
                scoresList = [list(map(float, errorMAE_Train)),
                              list(map(float, errorMSE_Train)),
                              list(map(float, errorMaxError_Train)),
                              list(map(float, errorMAE_Test)),
                              list(map(float, errorMSE_Test)),
                              list(map(float, errorMaxError_Test)),
                              ]
                scoresList = [item for sublist in scoresList for item in sublist]  # map sublist to fat list
                new_row = new_row + scoresList

                # Confirm file exists.
                # If not, create it, add headers, then append new data
                try:
                    wb = op.load_workbook(workbookFilePath)  # load the xlsx file
                    ws = wb.worksheets[0]  # select first worksheet
                except FileNotFoundError:
                    headers_row = ['Technique']
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('MAE-Tr-P' + str(i + 1))
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('MSE-Tr-P' + str(i + 1))
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('maxError-Tr-P' + str(i + 1))
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('MAE-Te-P' + str(i + 1))
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('MSE-Te-P' + str(i + 1))
                    for i in range(0, outputData_TrainVal_Shape[1]):
                        headers_row.append('maxError-Te-P' + str(i + 1))
                    wb = op.Workbook()
                    ws = wb.active
                    ws.append(headers_row)
                    wb.save(workbookFilePath)

                ws.append(new_row)
                wb.save(workbookFilePath)

        print(file_manip.getCurrentDatetimeForConsole() + "::Training Finished Successfully!")
        return listStr_ModelPaths, exportBaseDir, workbookDirPath

    def openModel(self, modelPath):
        pass

    def predict(self, x, y):
        dictModelPredictions = {}
        for _methodKey_ in self._MLR_dictMethods.keys():
            if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                print(_methodKey_ + ' predicts...')

                dictModelPredictions[_methodKey_] = {}
                dictModelPredictions[_methodKey_]['real'] = y
                dictModelPredictions[_methodKey_]['pred'] = np.array(
                    self._MLR_dictMethods[_methodKey_][MLR_KEY_TRAINED_MODEL].predict(x))

        return dictModelPredictions
