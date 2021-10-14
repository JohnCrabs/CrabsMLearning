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
    RandomizedSearchCV,
    cross_val_score
)

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
    ML_REG_GRADIENT_BOOSTING_REGRESSOR
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
    ML_SOLVER_LSQR,
    ML_SOLVER_SPARSE_CG,
    ML_SOLVER_SAG,
    ML_SOLVER_SAGA,
    ML_SOLVER_LBFGS
]

ML_KEY_METHOD = 'Method'
ML_KEY_PARAM_GRID = 'Grid Parameter'

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


# A class to store the Machine Learning Regression algorithms
class MachineLearningRegression:
    def __init__(self):
        self._MLR_dictMethods = {}

        self._MLR_KEY_MIN = 'min'
        self._MLR_KEY_MAX = 'max'
        self._MLR_KEY_STEP = 'step'
        self._MLR_KEY_STATE = 'state'

        self._MLR_RIDGE_ALPHA_DEFAULT = 1.0
        self._MLR_RIDGE_TOL_DEFAULT = [1e-3]
        self._MLR_RIDGE_SOLVER_DEFAULT = [ML_SOLVER_AUTO]

    def setML_dict(self):
        # for _method_ in ML_REG_METHODS:
        #     self._ML_dictMethods[_method_] = {}

        self._MLR_dictMethods[ML_REG_LINEAR_REGRESSION] = {}
        self._MLR_dictMethods[ML_REG_RIDGE] = {}
        self._MLR_dictMethods[ML_REG_BAYESIAN_RIDGE] = {}
        self._MLR_dictMethods[ML_REG_LASSO] = {}
        self._MLR_dictMethods[ML_REG_LASSO_LARS] = {}
        self._MLR_dictMethods[ML_REG_TWEEDIE_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_SGD_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_SVR] = {}
        self._MLR_dictMethods[ML_REG_LINEAR_SVR] = {}
        self._MLR_dictMethods[ML_REG_NEAREST_NEIGHBORS] = {}
        self._MLR_dictMethods[ML_REG_K_NEIGHBORS_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_DECISION_TREE_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_RANDOM_FOREST_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_ADA_BOOST_REGRESSOR] = {}
        self._MLR_dictMethods[ML_REG_GRADIENT_BOOSTING_REGRESSOR] = {}

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

    # ********************************** #
    # ***** RESTORE DEFAULT VALUES ***** #
    # ********************************** #

    def restore_LinearRegression_Defaults(self):
        self._MLR_dictMethods[ML_REG_LINEAR_REGRESSION] = {ML_KEY_METHOD: LinearRegression(),
                                                           self._MLR_KEY_STATE: False,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_Ridge_Defaults(self):
        self._MLR_dictMethods[ML_REG_RIDGE] = {ML_KEY_METHOD: Ridge(),
                                               self._MLR_KEY_STATE: False,
                                               ML_KEY_PARAM_GRID: {
                                                   ML_KEY_ALPHA: {self._MLR_KEY_MIN: self._MLR_RIDGE_ALPHA_DEFAULT,
                                                                  self._MLR_KEY_MAX: self._MLR_RIDGE_ALPHA_DEFAULT,
                                                                  self._MLR_KEY_STEP: self._MLR_RIDGE_ALPHA_DEFAULT},
                                                   ML_KEY_TOL: self._MLR_RIDGE_TOL_DEFAULT,
                                                   ML_KEY_SOLVER: self._MLR_RIDGE_SOLVER_DEFAULT
                                               }
                                               }

    def restore_BayesianRidge_Default(self):
        self._MLR_dictMethods[ML_REG_BAYESIAN_RIDGE] = {ML_KEY_METHOD: BayesianRidge(),
                                                        self._MLR_KEY_STATE: False,
                                                        ML_KEY_PARAM_GRID: {}
                                                        }

    def restore_Lasso_Default(self):
        self._MLR_dictMethods[ML_REG_LASSO] = {ML_KEY_METHOD: Lasso(),
                                               self._MLR_KEY_STATE: False,
                                               ML_KEY_PARAM_GRID: {}
                                               }

    def restore_LassoLars_Default(self):
        self._MLR_dictMethods[ML_REG_LASSO_LARS] = {ML_KEY_METHOD: LassoLars(),
                                                    self._MLR_KEY_STATE: False,
                                                    ML_KEY_PARAM_GRID: {}
                                                    }

    def restore_TweedieRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_TWEEDIE_REGRESSOR] = {ML_KEY_METHOD: TweedieRegressor(),
                                                           self._MLR_KEY_STATE: False,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_SGDRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_SGD_REGRESSOR] = {ML_KEY_METHOD: SGDRegressor(),
                                                       self._MLR_KEY_STATE: False,
                                                       ML_KEY_PARAM_GRID: {}
                                                       }

    def restore_SVR_Default(self):
        self._MLR_dictMethods[ML_REG_SVR] = {ML_KEY_METHOD: SVR(),
                                             self._MLR_KEY_STATE: False,
                                             ML_KEY_PARAM_GRID: {}
                                             }

    def restore_LinearSVR_Default(self):
        self._MLR_dictMethods[ML_REG_LINEAR_SVR] = {ML_KEY_METHOD: LinearSVR(),
                                                    self._MLR_KEY_STATE: False,
                                                    ML_KEY_PARAM_GRID: {}
                                                    }

    def restore_NearestNeighbor_Default(self):
        self._MLR_dictMethods[ML_REG_NEAREST_NEIGHBORS] = {ML_KEY_METHOD: NearestNeighbors(),
                                                           self._MLR_KEY_STATE: False,
                                                           ML_KEY_PARAM_GRID: {}
                                                           }

    def restore_KNeighborsRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_K_NEIGHBORS_REGRESSOR] = {ML_KEY_METHOD: KNeighborsRegressor(),
                                                               self._MLR_KEY_STATE: False,
                                                               ML_KEY_PARAM_GRID: {}
                                                               }

    def restore_DecisionTreeRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_DECISION_TREE_REGRESSOR] = {ML_KEY_METHOD: DecisionTreeRegressor(),
                                                                 self._MLR_KEY_STATE: False,
                                                                 ML_KEY_PARAM_GRID: {}
                                                                 }

    def restore_RandomForestRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_RANDOM_FOREST_REGRESSOR] = {ML_KEY_METHOD: RandomForestRegressor(),
                                                                 self._MLR_KEY_STATE: False,
                                                                 ML_KEY_PARAM_GRID: {}
                                                                 }

    def restore_AdaBoostRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_ADA_BOOST_REGRESSOR] = {ML_KEY_METHOD: AdaBoostRegressor(),
                                                             self._MLR_KEY_STATE: False,
                                                             ML_KEY_PARAM_GRID: {}
                                                             }

    def restore_GradientBoostingRegressor_Default(self):
        self._MLR_dictMethods[ML_REG_GRADIENT_BOOSTING_REGRESSOR] = {ML_KEY_METHOD: GradientBoostingRegressor(),
                                                                     self._MLR_KEY_STATE: False,
                                                                     ML_KEY_PARAM_GRID: {}
                                                                     }

    # ***************************** #
    # ***** SETTERS / GETTERS ***** #
    # ***************************** #

    # ****** RIDGE ***** #
    def setRidge_alphaMin(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_MIN] = value

    def setRidge_alphaMax(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_MAX] = value

    def setRidge_alphaStep(self, value: float):
        self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_STEP] = value

    def getRidge_alphaMin(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_MIN]

    def getRidge_alphaMax(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_MAX]

    def getRidge_alphaStep(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][ML_KEY_PARAM_GRID][ML_KEY_ALPHA][self._MLR_KEY_STEP]

    def getRidge_alphaMin_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT

    def getRidge_alphaMax_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT

    def getRidge_alphaStep_Default(self):
        return self._MLR_RIDGE_ALPHA_DEFAULT

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

    def setRidge_state(self, value: bool):
        self._MLR_dictMethods[ML_REG_RIDGE][self._MLR_KEY_STATE] = value

    def getRidge_state(self):
        return self._MLR_dictMethods[ML_REG_RIDGE][self._MLR_KEY_STATE]
