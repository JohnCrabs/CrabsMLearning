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
    KDTree,
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
ML_REG_KDTREE = 'KDTree'
ML_REG_K_NEIGHBORS_REGRESSOR = 'KNeighborsRegressor'

ML_REG_DECISION_TREE_REGRESSOR = 'DecisionTreeRegressor'

ML_REG_RANDOM_FOREST_REGRESSOR = 'RandomForestRegressor'
ML_REG_ADA_BOOST_REGRESSOR = 'AdaBoostRegressor'
ML_REG_GRADIENT_BOOSTING_REGRESSOR = 'GradientBoostingRegressor'

ML_REG_METHODS = [ML_REG_LINEAR_REGRESSION,
                  ML_REG_RIDGE,
                  ML_REG_BAYESIAN_RIDGE,
                  ML_REG_LASSO,
                  ML_REG_LASSO_LARS,
                  ML_REG_TWEEDIE_REGRESSOR,
                  ML_REG_SGD_REGRESSOR,
                  ML_REG_SVR,
                  ML_REG_LINEAR_SVR,
                  ML_REG_NEAREST_NEIGHBORS,
                  ML_REG_KDTREE,
                  ML_REG_K_NEIGHBORS_REGRESSOR,
                  ML_REG_DECISION_TREE_REGRESSOR,
                  ML_REG_RANDOM_FOREST_REGRESSOR,
                  ML_REG_ADA_BOOST_REGRESSOR,
                  ML_REG_GRADIENT_BOOSTING_REGRESSOR
                  ]


# A class to store the Machine Learning Regression algorithms
class MachineLearningRegression:
    def __init__(self):
        self._ML_dictMethods = {}

    def setML_dict(self):
        for _method_ in ML_REG_METHODS:
            self._ML_dictMethods[_method_] = {}

