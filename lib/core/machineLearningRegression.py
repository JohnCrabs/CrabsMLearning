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

###########################################
# ***** MACHINE LEARNING TECHNIQUES ***** #
#                                         #

MLR_REG_LINEAR_REGRESSION = 'LinearRegression'
MLR_REG_RIDGE = 'Ridge'
MLR_REG_BAYESIAN_RIDGE = 'BayesianRidge'
MLR_REG_LASSO = 'Lasso'
MLR_REG_LASSO_LARS = 'LassoLars'
MLR_REG_TWEEDIE_REGRESSOR = 'TweedieRegressor'
MLR_REG_SGD_REGRESSOR = 'SGDRegressor'

MLR_REG_SVR = 'SVR'
MLR_REG_LINEAR_SVR = 'LinearSVR'

MLR_REG_NEAREST_NEIGHBORS = 'NearestNeighbors'
MLR_REG_K_NEIGHBORS_REGRESSOR = 'KNeighborsRegressor'

MLR_REG_DECISION_TREE_REGRESSOR = 'DecisionTreeRegressor'

MLR_REG_RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'
MLR_REG_ADA_BOOST_REGRESSOR = 'AdaBoostRegressor'
MLR_REG_GRADIENT_BOOSTING_REGRESSOR = 'GradientBoostingRegressor'

MLR_REG_COVID_DNN = 'Covid_DeepNeuralNetwork'
MLR_REG_COVID_LSTM = 'Covid_LongShortTermMemoryNeuralNetwork'
MLR_REG_COVID_RNN = 'Covid_RecurrentNeuralNetwork'
MLR_REG_COVID_SIMPLE_RNN = 'Covid_SimpleRecurrentNeuralNetwork'

MLR_REG_LSTM = 'LongShortTermMemoryNetwork'
MLR_REG_CNN = 'ConvolutionalNeuralNetwork'
MLR_REG_CUSTOM = 'CustomNeuralNetwork'

MLR_REG_METHODS = [
    MLR_REG_LINEAR_REGRESSION,
    MLR_REG_RIDGE,
    MLR_REG_BAYESIAN_RIDGE,
    MLR_REG_LASSO,
    MLR_REG_LASSO_LARS,
    MLR_REG_TWEEDIE_REGRESSOR,
    MLR_REG_SGD_REGRESSOR,
    MLR_REG_SVR,
    MLR_REG_LINEAR_SVR,
    MLR_REG_NEAREST_NEIGHBORS,
    MLR_REG_K_NEIGHBORS_REGRESSOR,
    MLR_REG_DECISION_TREE_REGRESSOR,
    MLR_REG_RANDOM_FOREST_REGRESSOR,
    MLR_REG_ADA_BOOST_REGRESSOR,
    MLR_REG_GRADIENT_BOOSTING_REGRESSOR,
    MLR_REG_COVID_DNN,
    MLR_REG_COVID_LSTM,
    MLR_REG_COVID_RNN,
    MLR_REG_COVID_SIMPLE_RNN
]

_MLR_NO_TUNING_LIST = [
    MLR_REG_LINEAR_REGRESSION
]

_MLR_TUNING_NON_DEEP_METHODS = [
    MLR_REG_RIDGE,
    MLR_REG_BAYESIAN_RIDGE,
    MLR_REG_LASSO,
    MLR_REG_LASSO_LARS,
    MLR_REG_TWEEDIE_REGRESSOR,
    MLR_REG_SGD_REGRESSOR,
    MLR_REG_SVR,
    MLR_REG_LINEAR_SVR,
    MLR_REG_NEAREST_NEIGHBORS,
    MLR_REG_K_NEIGHBORS_REGRESSOR,
    MLR_REG_DECISION_TREE_REGRESSOR,
    MLR_REG_RANDOM_FOREST_REGRESSOR,
    MLR_REG_ADA_BOOST_REGRESSOR,
    MLR_REG_GRADIENT_BOOSTING_REGRESSOR
]

_MLR_TUNING_DEEP_METHODS = [
    MLR_REG_COVID_DNN,
    MLR_REG_COVID_LSTM,
    MLR_REG_COVID_RNN,
    MLR_REG_COVID_SIMPLE_RNN
]

_MLR_3RD_DIM_DEEP_METHODS = [
    MLR_REG_COVID_LSTM,
    MLR_REG_COVID_RNN,
    MLR_REG_COVID_SIMPLE_RNN
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
DMLR_EPOCHS = 100


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
        for _method_ in MLR_REG_METHODS:
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
        self.restore_Covid_LongShortTermMemoryNetworkRegressor_Default()
        self.restore_Covid_RecurrentNeuralNetworkRegressor_Default()
        self.restore_Covid_SimpleRecurrentNeuralNetworkRegressor_Default()

    # ********************************** #
    # ***** RESTORE DEFAULT VALUES ***** #
    # ********************************** #

    def restore_LinearRegression_Defaults(self):
        self._MLR_dictMethods[MLR_REG_LINEAR_REGRESSION] = {MLR_KEY_METHOD: LinearRegression(),
                                                            self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                            MLR_KEY_PARAM_GRID: {},
                                                            MLR_KEY_TRAINED_MODEL: None
                                                            }

    def restore_Ridge_Defaults(self):
        self._MLR_dictMethods[MLR_REG_RIDGE] = {MLR_KEY_METHOD: Ridge(),
                                                self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                MLR_KEY_CUSTOM_PARAM: {
                                                    MLR_KEY_ALPHA: {
                                                        self._MLR_KEY_MIN: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE,
                                                        self._MLR_KEY_MAX: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE,
                                                        self._MLR_KEY_STEP: self._MLR_RIDGE_ALPHA_DEFAULT_VALUE
                                                    }
                                                },
                                                MLR_KEY_PARAM_GRID: {
                                                    MLR_KEY_ALPHA: self._MLR_RIDGE_ALPHA_DEFAULT,
                                                    MLR_KEY_TOL: self._MLR_RIDGE_TOL_DEFAULT,
                                                    MLR_KEY_SOLVER: self._MLR_RIDGE_SOLVER_DEFAULT
                                                },
                                                MLR_KEY_TRAINED_MODEL: None
                                                }

    def restore_BayesianRidge_Default(self):
        self._MLR_dictMethods[MLR_REG_BAYESIAN_RIDGE] = {MLR_KEY_METHOD: BayesianRidge(),
                                                         self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                         MLR_KEY_PARAM_GRID: {}
                                                         }

    def restore_Lasso_Default(self):
        self._MLR_dictMethods[MLR_REG_LASSO] = {MLR_KEY_METHOD: Lasso(),
                                                self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                MLR_KEY_PARAM_GRID: {}
                                                }

    def restore_LassoLars_Default(self):
        self._MLR_dictMethods[MLR_REG_LASSO_LARS] = {MLR_KEY_METHOD: LassoLars(),
                                                     self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                     MLR_KEY_PARAM_GRID: {}
                                                     }

    def restore_TweedieRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_TWEEDIE_REGRESSOR] = {MLR_KEY_METHOD: TweedieRegressor(),
                                                            self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                            MLR_KEY_PARAM_GRID: {}
                                                            }

    def restore_SGDRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_SGD_REGRESSOR] = {MLR_KEY_METHOD: SGDRegressor(),
                                                        self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                        MLR_KEY_PARAM_GRID: {}
                                                        }

    def restore_SVR_Default(self):
        self._MLR_dictMethods[MLR_REG_SVR] = {MLR_KEY_METHOD: SVR(),
                                              self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                              MLR_KEY_PARAM_GRID: {
                                                  MLR_KEY_KERNEL: self._MLR_SVR_KERNEL_DEFAULT,
                                                  MLR_KEY_DEGREE: self._MLR_SVR_DEGREE_DEFAULT,
                                                  MLR_KEY_GAMMA: self._MLR_SVR_GAMMA_DEFAULT,
                                                  MLR_KEY_TOL: self._MLR_SVR_TOL_DEFAULT
                                              }
                                              }

    def restore_LinearSVR_Default(self):
        self._MLR_dictMethods[MLR_REG_LINEAR_SVR] = {MLR_KEY_METHOD: LinearSVR(),
                                                     self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                     MLR_KEY_PARAM_GRID: {}
                                                     }

    def restore_NearestNeighbor_Default(self):
        self._MLR_dictMethods[MLR_REG_NEAREST_NEIGHBORS] = {MLR_KEY_METHOD: NearestNeighbors(),
                                                            self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                            MLR_KEY_PARAM_GRID: {}
                                                            }

    def restore_KNeighborsRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_K_NEIGHBORS_REGRESSOR] = {MLR_KEY_METHOD: KNeighborsRegressor(),
                                                                self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                                MLR_KEY_PARAM_GRID: {}
                                                                }

    def restore_DecisionTreeRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_DECISION_TREE_REGRESSOR] = {MLR_KEY_METHOD: DecisionTreeRegressor(),
                                                                  self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                                  MLR_KEY_PARAM_GRID: {}
                                                                  }

    def restore_RandomForestRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_RANDOM_FOREST_REGRESSOR] = {MLR_KEY_METHOD: RandomForestRegressor(),
                                                                  self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                                  MLR_KEY_PARAM_GRID: {}
                                                                  }

    def restore_AdaBoostRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_ADA_BOOST_REGRESSOR] = {MLR_KEY_METHOD: AdaBoostRegressor(),
                                                              self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                              MLR_KEY_PARAM_GRID: {}
                                                              }

    def restore_GradientBoostingRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_GRADIENT_BOOSTING_REGRESSOR] = {MLR_KEY_METHOD: GradientBoostingRegressor(),
                                                                      self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                                      MLR_KEY_PARAM_GRID: {}
                                                                      }

    def restore_Covid_DeepNeuralNetworkRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_COVID_DNN] = {MLR_KEY_METHOD: self.DeepLearning_Covid_DNN,
                                                    self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                    MLR_KEY_PARAM_GRID: {
                                                        MLR_KEY_ACTIVATION_FUNCTION: DMLR_ACTIVATION_FUNCTIONS,
                                                        MLR_KEY_NUMBER_OF_EPOCHS: DMLR_EPOCHS
                                                    },
                                                    MLR_KEY_TRAINED_MODEL: None
                                                    }

    def restore_Covid_LongShortTermMemoryNetworkRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_COVID_LSTM] = {MLR_KEY_METHOD: self.DeepLearning_Covid_LSTM,
                                                     self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                     MLR_KEY_PARAM_GRID: {
                                                         MLR_KEY_ACTIVATION_FUNCTION: DMLR_ACTIVATION_FUNCTIONS,
                                                         MLR_KEY_NUMBER_OF_EPOCHS: DMLR_EPOCHS
                                                     },
                                                     MLR_KEY_TRAINED_MODEL: None,
                                                     MLR_KEY_3RD_DIM_SIZE: 1
                                                     }

    def restore_Covid_RecurrentNeuralNetworkRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_COVID_RNN] = {MLR_KEY_METHOD: self.DeepLearning_Covid_RNN,
                                                    self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                    MLR_KEY_PARAM_GRID: {
                                                        MLR_KEY_ACTIVATION_FUNCTION: DMLR_ACTIVATION_FUNCTIONS,
                                                        MLR_KEY_NUMBER_OF_EPOCHS: DMLR_EPOCHS
                                                    },
                                                    MLR_KEY_TRAINED_MODEL: None,
                                                    MLR_KEY_3RD_DIM_SIZE: 1
                                                    }

    def restore_Covid_SimpleRecurrentNeuralNetworkRegressor_Default(self):
        self._MLR_dictMethods[MLR_REG_COVID_SIMPLE_RNN] = {MLR_KEY_METHOD: self.DeepLearning_Covid_SimpleRNN,
                                                           self._MLR_KEY_STATE: MLR_EXEC_STATE,
                                                           MLR_KEY_PARAM_GRID: {
                                                               MLR_KEY_ACTIVATION_FUNCTION: DMLR_ACTIVATION_FUNCTIONS,
                                                               MLR_KEY_NUMBER_OF_EPOCHS: DMLR_EPOCHS
                                                           },
                                                           MLR_KEY_TRAINED_MODEL: None,
                                                           MLR_KEY_3RD_DIM_SIZE: 1
                                                           }

    # ********************************* #
    # ***** DEEP LEARNING METHODS ***** #
    # ********************************* #
    @staticmethod
    def DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model, directory, name,
                         epochs=100, tuner_objective='loss', early_stop_monitor='val_loss'):
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

        stop_early = keras.callbacks.EarlyStopping(monitor=early_stop_monitor, patience=5)
        tuner.search(train_x, train_y, epochs=epochs, validation_split=0.2, callbacks=[stop_early])

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

    def DeepLearning_Covid_DNN(self, train_x, train_y, test_x, test_y, exportDirectory,
                               epochs: int, activation_function_list: []):
        inputSize = train_x.shape[1]
        outputSize = train_y.shape[1]

        def ffunc_build_model(hp):
            # Create Model - Sequential
            ffunc_model = keras.Sequential()
            # Add Input Later
            ffunc_model.add(keras.Input(shape=(inputSize,)))
            # Add First Hidden Layer - l1
            ffunc_model.add(keras.layers.Dense(inputSize,
                                               activation=hp.Choice('activation_l1',
                                                                    values=activation_function_list)))
            # Add Dropout Layer - d1
            ffunc_model.add(keras.layers.Dropout(.2, input_shape=(2,)))
            # Add Second Hidden Layer - l2
            ffunc_model.add(keras.layers.Dense(int(inputSize / 2) + 2,
                                               activation=hp.Choice('activation_l2',
                                                                    values=activation_function_list)))
            # Add Dropout Layer - d2
            ffunc_model.add(keras.layers.Dropout(.2, input_shape=(2,)))
            # Add Third Hidden Layer - l3
            ffunc_model.add(keras.layers.Dense(inputSize,
                                               activation=hp.Choice('activation_l3',
                                                                    values=activation_function_list)))
            # Add Dropout Layer - d3
            ffunc_model.add(keras.layers.Dropout(.2, input_shape=(2,)))
            # Add Output Layer
            ffunc_model.add(keras.layers.Dense(outputSize,
                                               activation=hp.Choice('activation_lo',
                                                                    values=activation_function_list)))
            hp_learning_rate = hp.Choice('learning_rate', values=[0.1, 0.01, 0.001, 0.0001])
            ffunc_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                                loss='mae')
            ffunc_model.summary()

            return ffunc_model

        model = self.DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model,
                                      exportDirectory, MLR_REG_COVID_DNN, epochs=epochs)
        # print(model)

        return model

    def DeepLearning_Covid_LSTM(self, train_x, train_y, test_x, test_y, exportDirectory,
                                epochs: int, activation_function_list: []):
        inputSize = train_x.shape[1]
        outputSize = train_y.shape[1]
        expandDimSize = self._MLR_dictMethods[MLR_REG_COVID_LSTM][MLR_KEY_3RD_DIM_SIZE]

        def ffunc_build_model(hp):
            # Create Model - Sequential
            ffunc_model = keras.Sequential()
            # Add Input Layer
            ffunc_model.add(keras.Input(shape=(inputSize,)))
            # Add Reshape Layer
            ffunc_model.add(
                keras.layers.Reshape(target_shape=(expandDimSize, int(inputSize / expandDimSize),)
                                     ))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(inputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_conv_l1',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - LSTM
            ffunc_model.add(
                keras.layers.LSTM(int(inputSize / expandDimSize), return_sequences=True,
                                  activation=hp.Choice('activation_lstm_l2',
                                                       values=activation_function_list)
                                  ))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(inputSize / expandDimSize) * 2, 1,
                                    activation=hp.Choice('activation_conv_l3',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - LSTM
            ffunc_model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(int(inputSize / expandDimSize) * 2, return_sequences=True,
                                  activation=hp.Choice('activation_lstm_l4',
                                                       values=activation_function_list)
                                  )))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(outputSize / expandDimSize) * 2, 1,
                                    activation=hp.Choice('activation_conv_l5',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - LSTM
            ffunc_model.add(keras.layers.Bidirectional(
                keras.layers.LSTM(int(outputSize / expandDimSize) * 2, return_sequences=True,
                                  activation=hp.Choice('activation_lstm_l6',
                                                       values=activation_function_list)
                                  )))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(outputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_conv_l7',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - LSTM
            ffunc_model.add(
                keras.layers.LSTM(int(outputSize / expandDimSize), return_sequences=True,
                                  activation=hp.Choice('activation_lstm_l9',
                                                       values=activation_function_list)
                                  ))
            # Add Reshape Layer
            ffunc_model.add(
                keras.layers.Reshape(target_shape=(outputSize,)
                                     ))
            # Add Output Layer
            ffunc_model.add(
                keras.layers.Dense(outputSize, activation=hp.Choice('activation_lo',
                                                                    values=activation_function_list)
                                   ))

            hp_learning_rate = hp.Choice('learning_rate', values=[0.001])
            ffunc_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                                loss='mae')
            ffunc_model.summary()

            return ffunc_model

        model = self.DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model,
                                      exportDirectory, MLR_REG_COVID_LSTM, epochs=epochs)
        # print(model)

        return model

    def DeepLearning_Covid_RNN(self, train_x, train_y, test_x, test_y, exportDirectory,
                               epochs: int, activation_function_list: []):
        inputSize = train_x.shape[1]
        outputSize = train_y.shape[1]
        expandDimSize = self._MLR_dictMethods[MLR_REG_COVID_LSTM][MLR_KEY_3RD_DIM_SIZE]

        def ffunc_build_model(hp):
            # Create Model - Sequential
            ffunc_model = keras.Sequential()
            # Add Input Layer
            ffunc_model.add(keras.Input(shape=(inputSize,)))
            # Add Reshape Layer
            ffunc_model.add(keras.layers.Reshape(target_shape=(expandDimSize, int(inputSize / expandDimSize),)))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(inputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_l1',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - SimpleLSTM
            ffunc_model.add(
                keras.layers.RNN(int(inputSize / expandDimSize), return_sequences=True,
                                 activation=hp.Choice('activation_l2',
                                                      values=activation_function_list)
                                 ))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(outputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_l3',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - SimpleLSTM
            ffunc_model.add(
                keras.layers.RNN(int(outputSize / expandDimSize), return_sequences=True,
                                 activation=hp.Choice('activation_l4',
                                                      values=activation_function_list)
                                 ))
            # Add Reshape Layer
            ffunc_model.add(keras.layers.Reshape(target_shape=(outputSize,)))
            # Add Output Layer
            ffunc_model.add(keras.layers.Dense(outputSize, activation=hp.Choice('activation_lo',
                                                                                values=activation_function_list)))

            hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])
            ffunc_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                                loss='mae')
            ffunc_model.summary()

            return ffunc_model

        model = self.DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model,
                                      exportDirectory, MLR_REG_COVID_RNN, epochs=epochs)
        # print(model)

        return model

    def DeepLearning_Covid_SimpleRNN(self, train_x, train_y, test_x, test_y, exportDirectory,
                                     epochs: int, activation_function_list: []):
        inputSize = train_x.shape[1]
        outputSize = train_y.shape[1]
        expandDimSize = self._MLR_dictMethods[MLR_REG_COVID_LSTM][MLR_KEY_3RD_DIM_SIZE]

        def ffunc_build_model(hp):
            # Create Model - Sequential
            ffunc_model = keras.Sequential()
            # Add Input Layer
            ffunc_model.add(keras.Input(shape=(inputSize,)))
            # Add Reshape Layer
            ffunc_model.add(keras.layers.Reshape(target_shape=(expandDimSize, int(inputSize / expandDimSize),)))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(inputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_conv_l1',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - SimpleRNN
            ffunc_model.add(
                keras.layers.SimpleRNN(int(inputSize / expandDimSize), return_sequences=True,
                                       activation=hp.Choice('activation_lstm_l2',
                                                            values=activation_function_list)
                                       ))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(inputSize / expandDimSize) * 2, 1,
                                    activation=hp.Choice('activation_conv_l3',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - LSTM
            ffunc_model.add(keras.layers.Bidirectional(
                keras.layers.SimpleRNN(int(inputSize / expandDimSize) * 2, return_sequences=True,
                                       activation=hp.Choice('activation_lstm_l4',
                                                            values=activation_function_list)
                                       )))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(outputSize / expandDimSize) * 2, 1,
                                    activation=hp.Choice('activation_conv_l5',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - SimpleRNN
            ffunc_model.add(keras.layers.Bidirectional(
                keras.layers.SimpleRNN(int(outputSize / expandDimSize) * 2, return_sequences=True,
                                       activation=hp.Choice('activation_lstm_l6',
                                                            values=activation_function_list)
                                       )))
            # Add Hidden Layer - Conv1D
            ffunc_model.add(
                keras.layers.Conv1D(int(outputSize / expandDimSize), 1,
                                    activation=hp.Choice('activation_conv_l7',
                                                         values=activation_function_list)
                                    ))
            # Add Hidden Layer - SimpleRNN
            ffunc_model.add(
                keras.layers.SimpleRNN(int(outputSize / expandDimSize), return_sequences=True,
                                       activation=hp.Choice('activation_lstm_l9',
                                                            values=activation_function_list)
                                       ))
            # Add Reshape Layer
            ffunc_model.add(
                keras.layers.Reshape(target_shape=(outputSize,)
                                     ))
            # Add Output Layer
            ffunc_model.add(
                keras.layers.Dense(outputSize, activation=hp.Choice('activation_lo',
                                                                    values=activation_function_list)
                                   ))
            # Add Reshape Layer
            ffunc_model.add(keras.layers.Reshape(target_shape=(outputSize,)))
            # Add Output Layer
            ffunc_model.add(keras.layers.Dense(outputSize, activation=hp.Choice('activation_lo',
                                                                                values=activation_function_list)))

            hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001])
            ffunc_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                                loss='mae')
            ffunc_model.summary()

            return ffunc_model

        model = self.DeepLearning_fit(train_x, train_y, test_x, test_y, ffunc_build_model,
                                      exportDirectory, MLR_REG_COVID_SIMPLE_RNN, epochs=epochs)
        # print(model)

        return model

    # ***************************** #
    # ***** SETTERS / GETTERS ***** #
    # ***************************** #
    # ****** GLOBAL SETTINGS ***** #
    def set3rdDimensionSizeToDeepLearningMethods(self, dimSize: int):
        for _method_ in _MLR_3RD_DIM_DEEP_METHODS:
            self._MLR_dictMethods[_method_][MLR_KEY_3RD_DIM_SIZE] = dimSize

    # ****** LINEAR_REGRESSION ***** #
    def setLinearRegression_sate(self, state: bool):
        self._MLR_dictMethods[MLR_REG_LINEAR_REGRESSION][self._MLR_KEY_STATE] = state

    def getLinearRegression_sate(self):
        return self._MLR_dictMethods[MLR_REG_LINEAR_REGRESSION][self._MLR_KEY_STATE]

    # ****** RIDGE ***** #
    def setRidge_alphaMin(self, value: float):
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_MIN] = value
        self._setRidge_Alpha()

    def setRidge_alphaMax(self, value: float):
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_MAX] = value
        self._setRidge_Alpha()

    def setRidge_alphaStep(self, value: float):
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_STEP] = value
        self._setRidge_Alpha()

    def getRidge_alphaMin(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_MIN]

    def getRidge_alphaMax(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_MAX]

    def getRidge_alphaStep(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_CUSTOM_PARAM][MLR_KEY_ALPHA][self._MLR_KEY_STEP]

    def _setRidge_Alpha(self):
        stepVal = self.getRidge_alphaStep()
        minVal = self.getRidge_alphaMin()
        maxVal = self.getRidge_alphaMax() + stepVal
        tabValue = []
        for _value_ in np.arange(minVal, maxVal, stepVal):
            tabValue.append(_value_)
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_ALPHA] = tabValue

    def getRidge_Alpha(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_ALPHA]

    def getRidge_alphaMin_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def getRidge_alphaMax_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def getRidge_alphaStep_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT_VALUE

    def setRidge_Tol(self, value: []):
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_TOL] = value

    def getRidge_Tol(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_TOL]

    def getRidge_Tol_Default(self):
        return self._MLR_RIDGE_TOL_DEFAULT

    def setRidge_Solver(self, value: []):
        self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_SOLVER] = value

    def getRidge_Solver(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][MLR_KEY_PARAM_GRID][MLR_KEY_SOLVER]

    def getRidge_Solver_Default(self):
        return self._MLR_RIDGE_SOLVER_DEFAULT

    def setRidge_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_RIDGE][self._MLR_KEY_STATE] = state

    def getRidge_state(self):
        return self._MLR_dictMethods[MLR_REG_RIDGE][self._MLR_KEY_STATE]

    # ****** Bayesian Ridge ***** #
    def setBayesianRidge_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_BAYESIAN_RIDGE][self._MLR_KEY_STATE] = state

    def getBayesianRidge_state(self):
        return self._MLR_dictMethods[MLR_REG_BAYESIAN_RIDGE][self._MLR_KEY_STATE]

    # ****** Lasso ***** #
    def setLasso_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_LASSO][self._MLR_KEY_STATE] = state

    def getLasso_state(self):
        return self._MLR_dictMethods[MLR_REG_LASSO][self._MLR_KEY_STATE]

    # ****** Lasso Lars ***** #
    def setLassoLars_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_LASSO_LARS][self._MLR_KEY_STATE] = state

    def getLassoLars_state(self):
        return self._MLR_dictMethods[MLR_REG_LASSO_LARS][self._MLR_KEY_STATE]

    # ****** Tweedie Regressor ***** #
    def setTweedieRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_TWEEDIE_REGRESSOR][self._MLR_KEY_STATE] = state

    def getTweedieRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_TWEEDIE_REGRESSOR][self._MLR_KEY_STATE]

    # ****** SGD Regressor ***** #
    def setSGDRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_SGD_REGRESSOR][self._MLR_KEY_STATE] = state

    def getSGDRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_SGD_REGRESSOR][self._MLR_KEY_STATE]

    # ****** SVR ***** #
    def setSVR_Kernel(self, value: []):
        self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_KERNEL] = value

    def getSVR_Kernel(self):
        return self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_KERNEL]

    def getSVR_Kernel_Default(self):
        return self._MLR_SVR_KERNEL_DEFAULT

    def setSVR_Degree(self, value: []):
        self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_DEGREE] = value

    def getSVR_Degree(self):
        return self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_DEGREE]

    def getSVR_Degree_Default(self):
        return self._MLR_SVR_DEGREE_DEFAULT

    def setSVR_Gamma(self, value: []):
        self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_GAMMA] = value

    def getSVR_Gamma(self):
        return self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_GAMMA]

    def getSVR_Gamma_Default(self):
        return self._MLR_SVR_GAMMA_DEFAULT

    def setSVR_Tol(self, value: []):
        self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_TOL] = value

    def getSVR_Tol(self):
        return self._MLR_dictMethods[MLR_REG_SVR][MLR_KEY_PARAM_GRID][MLR_KEY_TOL]

    def getSVR_Tol_Default(self):
        return self._MLR_SVR_TOL_DEFAULT

    def setSVR_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_SVR][self._MLR_KEY_STATE] = state

    def getSVR_state(self):
        return self._MLR_dictMethods[MLR_REG_SVR][self._MLR_KEY_STATE]

    # ****** Linear SVR ***** #
    def setLinearSVR_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_LINEAR_SVR][self._MLR_KEY_STATE] = state

    def getLinearSVR_state(self):
        return self._MLR_dictMethods[MLR_REG_LINEAR_SVR][self._MLR_KEY_STATE]

    # ****** Nearest Neighbor ***** #
    def setNearestNeighbor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_NEAREST_NEIGHBORS][self._MLR_KEY_STATE] = state

    def getNearestNeighbor_state(self):
        return self._MLR_dictMethods[MLR_REG_NEAREST_NEIGHBORS][self._MLR_KEY_STATE]

    # ****** K Neighbors Regressor ***** #
    def setKNeighborsRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_K_NEIGHBORS_REGRESSOR][self._MLR_KEY_STATE] = state

    def getKNeighborsRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_K_NEIGHBORS_REGRESSOR][self._MLR_KEY_STATE]

    # ****** Decision Tree Regressor ***** #
    def setDecisionTreeRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_DECISION_TREE_REGRESSOR][self._MLR_KEY_STATE] = state

    def getDecisionTreeRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_DECISION_TREE_REGRESSOR][self._MLR_KEY_STATE]

    # ****** Random Forest Regressor ***** #
    def setRandomForestRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_RANDOM_FOREST_REGRESSOR][self._MLR_KEY_STATE] = state

    def getRandomForestRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_RANDOM_FOREST_REGRESSOR][self._MLR_KEY_STATE]

    # ****** Ada Boost Regressor ***** #
    def setAdaBoostRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_ADA_BOOST_REGRESSOR][self._MLR_KEY_STATE] = state

    def getAdaBoostRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_ADA_BOOST_REGRESSOR][self._MLR_KEY_STATE]

    # ****** Gradient Boosting Regressor ***** #
    def setGradientBoostingRegressor_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_GRADIENT_BOOSTING_REGRESSOR][self._MLR_KEY_STATE] = state

    def getGradientBoostingRegressor_state(self):
        return self._MLR_dictMethods[MLR_REG_GRADIENT_BOOSTING_REGRESSOR][self._MLR_KEY_STATE]

    # ****** Covid_DeepNeuralNetworkRegressor ***** #
    def setCovid_DNN_reg_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_COVID_DNN][self._MLR_KEY_STATE] = state

    def getCovid_DNN_reg_state(self):
        return self._MLR_dictMethods[MLR_REG_COVID_DNN][self._MLR_KEY_STATE]

    # ****** Covid_LongShortTermMemoryNeuralNetworkRegressor ***** #
    def setCovid_LSTM_reg_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_COVID_LSTM][self._MLR_KEY_STATE] = state

    def getCovid_LSTM_reg_state(self):
        return self._MLR_dictMethods[MLR_REG_COVID_LSTM][self._MLR_KEY_STATE]

    # ****** Covid_RecurrentNetworkRegressor ***** #
    def setCovid_RNN_reg_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_COVID_RNN][self._MLR_KEY_STATE] = state

    def getCovid_RNN_reg_state(self):
        return self._MLR_dictMethods[MLR_REG_COVID_RNN][self._MLR_KEY_STATE]

    # ****** Covid_SimpleRecurrentNetworkRegressor ***** #
    def setCovid_SimpleRNN_reg_state(self, state: bool):
        self._MLR_dictMethods[MLR_REG_COVID_SIMPLE_RNN][self._MLR_KEY_STATE] = state

    def getCovid_SimpleRNN_reg_state(self):
        return self._MLR_dictMethods[MLR_REG_COVID_SIMPLE_RNN][self._MLR_KEY_STATE]

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
                    activationFunctionList = self._MLR_dictMethods[_methodKey_][MLR_KEY_PARAM_GRID][
                        MLR_KEY_ACTIVATION_FUNCTION]

                    model = self._MLR_dictMethods[_methodKey_][MLR_KEY_METHOD](inputData_TrainVal,
                                                                               outputData_TrainVal,
                                                                               inputData_Test,
                                                                               outputData_Test,
                                                                               exportDeepLearningTunersPath,
                                                                               epochs,
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
