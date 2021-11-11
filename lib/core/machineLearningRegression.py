import os
import datetime as dt

import joblib
import numpy as np
import openpyxl as op

import lib.core.file_manipulation as file_manip

from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    BayesianRidge,
    Lasso,
    LassoLars,
    TweedieRegressor,
    SGDRegressor
)

from sklearn.svm import (
    SVR,
    LinearSVR
)

from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsRegressor
)

from sklearn.tree import (
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

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

ML_REG_LINEAR_REGRESSION = 'LinearRegression'
ML_REG_RIDGE = 'Ridge'
ML_REG_BAYESIAN_RIDGE = 'BayesianRidge'
ML_REG_LASSO = ' Lasso'
ML_REG_LASSO_LARS = 'LassoLars'
ML_REG_TWEEDIE_REGRESSOR = 'TweedieRegressor'
ML_REG_SGD_REGRESSOR = 'SGDRegressor'

ML_REG_SVR = 'SVR'
ML_REG_LINEAR_SVR = 'LinearSVR'

ML_REG_NEAREST_NEIGHBORS = 'NearestNeighbors'
ML_REG_K_NEIGHBORS_REGRESSOR = 'KNeighborsRegressor'

ML_REG_DECISION_TREE_REGRESSOR = 'DecisionTreeRegressor'

ML_REG_RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'
ML_REG_ADA_BOOST_REGRESSOR = 'AdaBoostRegressor'
ML_REG_GRADIENT_BOOSTING_REGRESSOR = 'GradientBoostingRegressor'

ML_REG_COVID_DNN = 'Covid_DeepNeuralNetwork'
ML_REG_LSTM = 'LongShortTermMemoryNetwork'
ML_REG_CNN = 'ConvolutionalNeuralNetwork'
ML_REG_CUSTOM = 'CustomNeuralNetwork'

ML_REG_METHODS = [
    ML_REG_LINEAR_REGRESSION,
    ML_REG_RIDGE,
    ML_REG_BAYESIAN_RIDGE,
    ML_REG_LASSO,
    ML_REG_LASSO_LARS,
    ML_REG_TWEEDIE_REGRESSOR,
    ML_REG_SGD_REGRESSOR,
    ML_REG_SVR,
    ML_REG_LINEAR_SVR,
    ML_REG_NEAREST_NEIGHBORS,
    ML_REG_K_NEIGHBORS_REGRESSOR,
    ML_REG_DECISION_TREE_REGRESSOR,
    ML_REG_RANDOM_FOREST_REGRESSOR,
    ML_REG_ADA_BOOST_REGRESSOR,
    ML_REG_GRADIENT_BOOSTING_REGRESSOR,
    ML_REG_COVID_DNN
]

ML_SOLVER_AUTO = 'auto'
ML_SOLVER_SVD = 'svd'
ML_SOLVER_CHOLESKY = 'cholesky'
ML_SOLVER_LSQR = 'lsqr'
ML_SOLVER_SPARSE_CG = 'sparse_cg'
ML_SOLVER_SAG = 'sag'
ML_SOLVER_SAGA = 'saga'
ML_SOLVER_LBFGS = 'lbfgs'

ML_SOLVER_OPTIONS = [
    ML_SOLVER_AUTO,
    ML_SOLVER_SVD,
    ML_SOLVER_CHOLESKY,
    # ML_SOLVER_LSQR,
    ML_SOLVER_SPARSE_CG,
    # ML_SOLVER_SAG,
    # ML_SOLVER_SAGA,
    # ML_SOLVER_LBFGS
]

ML_KEY_METHOD = 'Method'
ML_KEY_CUSTOM_PARAM = 'Custom Parameters'
ML_KEY_PARAM_GRID = 'Grid Parameter'
ML_KEY_TRAINED_MODEL = 'Trained Model'

ML_KEY_ALPHA = 'alpha'
ML_KEY_TOL = 'tol'
ML_KEY_SOLVER = 'solver'

ML_TOL_LIST = [
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

ML_EXEC_STATE = False

_ML_NO_TUNING_LIST = [
    ML_REG_LINEAR_REGRESSION
]

_ML_TUNING_NON_DEEP_METHODS = [
    ML_REG_RIDGE,
    ML_REG_BAYESIAN_RIDGE,
    ML_REG_LASSO,
    ML_REG_LASSO_LARS,
    ML_REG_TWEEDIE_REGRESSOR,
    ML_REG_SGD_REGRESSOR,
    ML_REG_SVR,
    ML_REG_LINEAR_SVR,
    ML_REG_NEAREST_NEIGHBORS,
    ML_REG_K_NEIGHBORS_REGRESSOR,
    ML_REG_DECISION_TREE_REGRESSOR,
    ML_REG_RANDOM_FOREST_REGRESSOR,
    ML_REG_ADA_BOOST_REGRESSOR,
    ML_REG_GRADIENT_BOOSTING_REGRESSOR
]

_ML_TUNING_DEEP_METHODS = [
    ML_REG_COVID_DNN,
    ML_REG_LSTM,
    ML_REG_CNN,
    ML_REG_CUSTOM
]


# A class to store the Machine Learning Regression algorithms
class MachineLearningRegression:
    def __init__(self):
        self._MLR_dictMethods = {}

        self._MLR_KEY_MIN = 'min'
        self._MLR_KEY_MAX = 'max'
        self._MLR_KEY_STEP = 'step'
        self._MLR_KEY_STATE = 'state'

        self._MLR_RIDGE_ALPHA_DEFAULT_VALUE = 1.0
        self._MLR_RIDGE_ALPHA_DEFAULT = [1.0]
        self._MLR_RIDGE_TOL_DEFAULT = [1e-3]
        self._MLR_RIDGE_SOLVER_DEFAULT = [ML_SOLVER_AUTO]

    def setML_dict(self):
        for _method_ in ML_REG_METHODS:
            self._MLR_dictMethods[_method_] = {}

        self.restore_LinearRegression_Defaults()
        self.restore_Ridge_Defaults()
        self.restore_BayesianRidge_Default()
        self.restore_Lasso_Default()
        self.restore_LassoLars_Default()
        self.restore_TweedieRegressor_Default()
        self.restore_SGDRegressor_Default()
        self.restore_SVR_Default()
        self.restore_LinearSVR_Default()
        self.restore_NearestNeighbor_Default()
        self.restore_KNeighborsRegressor_Default()
        self.restore_DecisionTreeRegressor_Default()
        self.restore_RandomForestRegressor_Default()
        self.restore_AdaBoostRegressor_Default()
        self.restore_GradientBoostingRegressor_Default()
        self.restore_Covid_DeepNeuralNetworkRegressor_Default()

    # ********************************** #
    # ***** RESTORE DEFAULT VALUES ***** #
    # ********************************** #

    def restore_LinearRegression_Defaults(self):
        self._MLR_dictMethods[ML_REG_LINEAR_REGRESSION] = {ML_KEY_METHOD: LinearRegression(),
                                                           self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_Ridge_Defaults(self):
        self._MLR_dictMethods[ML_REG_RIDGE] = {ML_KEY_METHOD: Ridge(),
                                               self._MLR_KEY_STATE: ML_EXEC_STATE,
                                               ML_KEY_CUSTOM_PARAM: {
                                                   ML_KEY_ALPHA: {
                                                       self._MLR_KEY_MIN: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE,
                                                       self._MLR_KEY_MAX: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE,
                                                       self._MLR_KEY_STEP: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE
                                                   }
                                               },
                                               ML_KEY_PARAM_GRID: {
                                                   ML_KEY_ALPHA: self._MLR_RIDGE_ALPHA_DEFAULT,
                                                   ML_KEY_TOL: self._MLR_RIDGE_TOL_DEFAULT,
                                                   ML_KEY_SOLVER: self._MLR_RIDGE_SOLVER_DEFAULT
                                               }
                                               }

    def restore_BayesianRidge_Default(self):
        self._MLR_dictMethods[ML_REG_BAYESIAN_RIDGE] = {ML_KEY_METHOD: BayesianRidge(),
                                                        self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                        ML_KEY_PARAM_GRID: {}
                                                        }

    def restore_Lasso_Default(self):
        self._MLR_dictMethods[ML_REG_LASSO] = {ML_KEY_METHOD: Lasso(),
                                               self._MLR_KEY_STATE: ML_EXEC_STATE,
                                               ML_KEY_PARAM_GRID: {}
                                               }

    def restore_LassoLars_Default(self):
        self._MLR_dictMethods[ML_REG_LASSO_LARS] = {ML_KEY_METHOD: LassoLars(),
                                                    self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                    ML_KEY_PARAM_GRID: {}
                                                    }

    def restore_TweedieRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_TWEEDIE_REGRESSOR] = {ML_KEY_METHOD: TweedieRegressor(),
                                                           self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_SGDRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_SGD_REGRESSOR] = {ML_KEY_METHOD: SGDRegressor(),
                                                       self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                       ML_KEY_PARAM_GRID: {}
                                                       }

    def restore_SVR_Default(self):
        self._MLR_dictMethods[ML_REG_SVR] = {ML_KEY_METHOD: SVR(),
                                             self._MLR_KEY_STATE: ML_EXEC_STATE,
                                             ML_KEY_PARAM_GRID: {}
                                             }

    def restore_LinearSVR_Default(self):
        self._MLR_dictMethods[ML_REG_LINEAR_SVR] = {ML_KEY_METHOD: LinearSVR(),
                                                    self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                    ML_KEY_PARAM_GRID: {}
                                                    }

    def restore_NearestNeighbor_Default(self):
        self._MLR_dictMethods[ML_REG_NEAREST_NEIGHBORS] = {ML_KEY_METHOD: NearestNeighbors(),
                                                           self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_KNeighborsRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_K_NEIGHBORS_REGRESSOR] = {ML_KEY_METHOD: KNeighborsRegressor(),
                                                               self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                               ML_KEY_PARAM_GRID: {}
                                                               }

    def restore_DecisionTreeRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_DECISION_TREE_REGRESSOR] = {ML_KEY_METHOD: DecisionTreeRegressor(),
                                                                 self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                                 ML_KEY_PARAM_GRID: {}
                                                                 }

    def restore_RandomForestRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_RANDOM_FOREST_REGRESSOR] = {ML_KEY_METHOD: RandomForestRegressor(),
                                                                 self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                                 ML_KEY_PARAM_GRID: {}
                                                                 }

    def restore_AdaBoostRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_ADA_BOOST_REGRESSOR] = {ML_KEY_METHOD: AdaBoostRegressor(),
                                                             self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                             ML_KEY_PARAM_GRID: {}
                                                             }

    def restore_GradientBoostingRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_GRADIENT_BOOSTING_REGRESSOR] = {ML_KEY_METHOD: GradientBoostingRegressor(),
                                                                     self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                                     ML_KEY_PARAM_GRID: {}
                                                                     }

    def restore_Covid_DeepNeuralNetworkRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_COVID_DNN] = {ML_KEY_METHOD: self.DeepLearning_Covid_DNN,
                                                   self._MLR_KEY_STATE: ML_EXEC_STATE,
                                                   ML_KEY_PARAM_GRID: {}
                                                   }

    # ********************************* #
    # ***** DEEP LEARNING METHODS ***** #
    # ********************************* #
    @staticmethod
    def DeepLearning_Covid_DNN(inputSize, outputSize):
        # DNN model here
        inputs = keras.Input(shape=(inputSize,))
        lr1 = keras.layers.Dense(inputSize * 2, activation='selu')(inputs)  # <-----------
        do1 = keras.layers.Dropout(0.2)(lr1)
        lr2 = keras.layers.Dense(inputSize, activation='selu')(do1)  # decoder  # <-----------
        lr3 = keras.layers.Dense(inputSize * 2, activation='selu')(lr2)  # <-----------
        do2 = keras.layers.Dropout(0.2)(lr3)
        outputs = keras.layers.Dense(outputSize, activation='sigmoid')(do2)  # <-----------
        DNN = keras.models.Model(inputs, outputs)
        DNN.compile(loss='mse', optimizer=keras.optimizers.RMSprop())  # <-----------

    # ***************************** #
    # ***** SETTERS / GETTERS ***** #
    # ***************************** #
    # ****** LINEAR_REGRESSION ***** #
    def setLinearRegression_sate(self, state: bool):
        self._MLR_dictMethods[ML_REG_LINEAR_REGRESSION][self._MLR_KEY_STATE] = state

    def getLinearRegression_sate(self):
        return self._MLR_dictMethods[ML_REG_LINEAR_REGRESSION][self._MLR_KEY_STATE]

    # ****** RIDGE ***** #
    def setRidge_alphaMin(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_MIN] = value
        self._setRidge_Alpha()

    def setRidge_alphaMax(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_MAX] = value
        self._setRidge_Alpha()

    def setRidge_alphaStep(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_STEP] = value
        self._setRidge_Alpha()

    def getRidge_alphaMin(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_MIN]

    def getRidge_alphaMax(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_MAX]

    def getRidge_alphaStep(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_CUSTOM_PARAM][ML_KEY_ALPHA][self._MLR_KEY_STEP]

    def _setRidge_Alpha(self):
        stepVal = self.getRidge_alphaStep()
        minVal = self.getRidge_alphaMin()
        maxVal = self.getRidge_alphaMax() + stepVal
        tabValue = []
        for _value_ in np.arange(minVal, maxVal, stepVal):
            tabValue.append(_value_)
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA] = tabValue

    def getRidge_Alpha(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA]

    def getRidge_alphaMin_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def getRidge_alphaMax_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def getRidge_alphaStep_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def setRidge_Tol(self, value: []):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_TOL] = value

    def getRidge_Tol(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_TOL]

    def getRidge_Tol_Default(self):
        return self._MLR_RIDGE_TOL_DEFAULT

    def setRidge_Solver(self, value: []):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_SOLVER] = value

    def getRidge_Solver(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_SOLVER]

    def getRidge_Solver_Default(self):
        return self._MLR_RIDGE_SOLVER_DEFAULT

    def setRidge_state(self, state: bool):
        self._MLR_dictMethods[ML_REG_RIDGE][self._MLR_KEY_STATE] = state

    def getRidge_state(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][self._MLR_KEY_STATE]

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
            if _methodKey_ in _ML_NO_TUNING_LIST:  # if method cannot be tuning (e.g. LinearRegression)
                if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                    print(file_manip.getCurrentDatetimeForConsole() + "::Training " + _methodKey_ + "...")  # console message
                    model = self._MLR_dictMethods[_methodKey_][ML_KEY_METHOD]
                    model.fit(inputData_TrainVal,
                              outputData_TrainVal)  # model.fit()
                    print(file_manip.getCurrentDatetimeForConsole() + "::...COMPLETED!")  # console message
                    # Export model
                    modelExportPath = os.path.normpath(exportTrainedModelsPath + modelName + '_' +
                                                       currentDatetime + H5_SUFFIX)
                    listStr_ModelPaths.append(modelExportPath)
                    joblib.dump(model, modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Model exported at: ", modelExportPath)
                    self._MLR_dictMethods[_methodKey_][ML_KEY_TRAINED_MODEL] = model

            elif _methodKey_ in _ML_TUNING_NON_DEEP_METHODS:  # elif method is not a tf.keras
                if self._MLR_dictMethods[_methodKey_][self._MLR_KEY_STATE]:
                    print(file_manip.getCurrentDatetimeForConsole() + "::Training " + _methodKey_ + "..")  # console message
                    # run Grid Search CV
                    model = GridSearchCV(self._MLR_dictMethods[_methodKey_][ML_KEY_METHOD],
                                         self._MLR_dictMethods[_methodKey_][ML_KEY_PARAM_GRID],
                                         n_jobs=-1)
                    model.fit(inputData_TrainVal,
                              outputData_TrainVal)  # model.fit()
                    print(file_manip.getCurrentDatetimeForConsole() + "::...COMPLETED!")  # console message
                    self._MLR_dictMethods[_methodKey_][ML_KEY_TRAINED_MODEL] = model.best_estimator_

                    # Export model
                    modelExportPath = os.path.normpath(exportTrainedModelsPath + modelName + '_' +
                                                       currentDatetime + H5_SUFFIX)
                    listStr_ModelPaths.append(modelExportPath)
                    joblib.dump(model, modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Model exported at: ", modelExportPath)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Best Estimator = ", model.best_estimator_)
                    print(file_manip.getCurrentDatetimeForConsole() + "::Best Score = ", model.best_score_)

            elif _methodKey_ in _ML_TUNING_DEEP_METHODS:  # elif method is keras
                pass
            else:  # else for security reasons only
                pass

            if model is not None:
                # if model is scikit learn model
                if model not in _ML_NO_TUNING_LIST:
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
                if _methodKey_ not in _ML_TUNING_DEEP_METHODS:
                    dictModelPredictions[_methodKey_] = {}
                    dictModelPredictions[_methodKey_]['real'] = y
                    dictModelPredictions[_methodKey_]['pred'] = np.array(
                        self._MLR_dictMethods[_methodKey_][ML_KEY_TRAINED_MODEL].predict(x))
        return dictModelPredictions
