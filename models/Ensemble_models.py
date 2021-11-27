# Importing packages

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging
import warnings

warnings.filterwarnings('ignore')

# Configuring logging operations

logging.basicConfig(filename='Ensemble_development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class Ensemble_models:
    """
    This class is used to build regression models using different ensemble techniques.
    Reference:
    reference 1 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html?highlight=adaboost%20regressor#sklearn.ensemble.AdaBoostRegressor
    reference 2 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradient%20boost%20regressor#sklearn.ensemble.GradientBoostingRegressor
    reference 3 - https://xgboost.readthedocs.io/en/latest/get_started.html
    reference 4 - https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

    Parameters:
    x_train: Training data frame containing the independent features.
    y_train: Training dataframe containing the dependent or target feature.
    x_test: Testing dataframe containing the independent features.
    y_test: Testing dataframe containing the dependent or target feature.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @property
    def adaboost_regressor(self):
        """Description: This method builds a model using AdaBoostRegressor algorithm, a type of ensemble technique imported from the sklearn library. It uses cross validation technique and chooses the best estimator with the best hyper parameters.
        Raises an exception if it fails

        returns: The Adaboost regressor models and prints the importance of each feature.
        """

        try:
            # instantiating the AdaBoostRegressor object
            ADB = AdaBoostRegressor()

            # parameter grid
            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                      'loss': ['linear', 'square', 'exponential']}

            # instantiating RandomizedSearchCV
            RCV = RandomizedSearchCV(estimator=ADB,
                                     param_distributions=params,
                                     n_iter=10,
                                     scoring='r2',
                                     n_jobs=-1,
                                     cv=10,
                                     verbose=5,
                                     random_state=42,
                                     return_train_score=True)

            print('Cross Validation process for the Adaboost regressor')

            # fitting on the train data
            RCV.fit(self.X_train, self.y_train)

            # displaying the best estimator
            print()
            print('The best estimator for the Adaboost regressor is', RCV.best_estimator_)

            # Feature importance by the Adaboost regressor

            ADB_feature_imp = pd.DataFrame(ADB.feature_importances_,
                                           index=self.X_train.columns,
                                           columns=['Feature_importance'])

            ADB_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)

            print()
            print(f'Feature Importance by the Adaboost regressor: {ADB_feature_imp}')
            print()

            # logging operation

            logging.info('Successfully built a model using Adaboost Regressor')

            logging.info('Exited the adaboost_regressor method of the Ensemble_models class')

            return ADB

        except Exception as e:

            # Logging operations
            logging.error(
                'Exception occurred in adaboost_regressor method of Ensemble_model class Exception message: ' + str(e))

            logging.info(
                'adaboost_regressor method unsuccessful. Exited the adaboost_regressor method of the Ensemble_models class')

    @property
    def gradientboosting_regressor(self):
        """Description: This method builds a model using GradientBoostingRegressor algorithm, a type of ensemble technique imported
           from the sklearn library. It uses cross validation technique and chooses the best estimator with the best hyper parameters.
           Raises an exception if it fails

           returns: The Gradient boosting regressor model and prints the importance of each feature
        """

        try:
            # instantiating the GradientBoostingRegressor object.
            GBR = GradientBoostingRegressor()

            # Parameter grid
            params = {'n_estimators': [5, 10, 15, 20, 40, 80, 100, 200],
                      'learning_rate': [0.001, 0.01, 0.2, 0.3, 0.5, 0.8, 1],
                      'loss': ['lr', 'lad', 'huber'],
                      'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                      'criterion': ['friedman_mse', 'mse'],
                      'min_samples_split': [2, 4, 8, 10],
                      'min_samples_leaf': [1, 10, 20, 50]}

            # instantiating RandomizedSearchCV
            RCV = RandomizedSearchCV(estimator=GBR,
                                     param_distributions=params,
                                     n_iter=10,
                                     scoring='r2',
                                     n_jobs=-1,
                                     cv=10,
                                     verbose=5,
                                     random_state=42,
                                     return_train_score=True)

            print('Cross Validation process for the Gradient Boosting Regressor')

            RCV.fit(self.X_train, self.y_train)
            print()

            # Displaying the best estimator
            print(f'The best estimator for the Gradient Boosting regressor is {RCV.best_estimator_}')

            # Building the best estimator recommended by the randomized search CV
            # as the final Gradient Boosting regressor.
            GBR = RCV.best_estimator_

            GBR.fit(self.X_train, self.y_train)

            GBR_feature_imp = pd.DataFrame(GBR.feature_importances_,
                                           index=self.X_train.columns,
                                           columns=['Feature_importance'])

            GBR_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)

            print()
            print(f'Feature Importance by the Gradient boosting regressor: {GBR_feature_imp}')
            print()

            # Logging operation
            logging.info('Successfully built a model using Gradient Boosting regressor')

            logging.info('Exited the gradientboosting_regressor method of the Ensemble_models class')

            return GBR

        except Exception as e:

            # logging operation
            logging.error(
                'Exception occurred in gradientboosting_regressor of the Ensemble_models class. Exception message: ' + str(
                    e))

            logging.info(
                'gradientboosting_regressor method unsuccessful. Exited the gradientboosting_regressor method of the Ensemble_models class')

    def xgb_regressor(self):
        """Description: This method builds a model using XGBRegressor algorithm, a type of ensemble technique imported from the
        xgboost library.It uses cross validation technique and chooses the best estimator with the best hyper parameters.
        Raises an exception if it fails
        returns: The XGBoost regressor model and prints the importance of each feature
        """
        try:
            # instantiating the XGBRegressor object
            XGBR = XGBRegressor()

            # Parameter grid
            params = {
                'learning_rate': [0.1, 0.2, 0.5, 0.8, 1],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 10],
                'subsample': [0.001, 0.009, 0.01, 0.09, 0.1, 0.4, 0.9, 1],
                'min_child_weight': [1, 2, 4, 5, 8],
                'gamma': [0.0, 0.1, 0.2, 0.3],
                'colsample_bytree': [0.3, 0.5, 0.7, 1.0, 1.4],
                'reg_alpha': [0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1, 4, 8, 10, 50, 100],
                'reg_lambda': [1, 4, 5, 10, 20, 50, 100, 200, 500, 800, 1000]}

            # instantiating RandomizedSearchCV
            RCV = RandomizedSearchCV(estimator=XGBR,
                                     param_distributions=params,
                                     n_iter=10,
                                     scoring='r2',
                                     cv=10,
                                     verbose=2,
                                     random_state=42,
                                     n_jobs=-1,
                                     return_train_score=True)

            print('Cross validation process for the XGBoost regressor')

            RCV.fit(self.X_train, self.y_train)  # Fitting on the train data
            print()

            # displaying the best estimator
            print(f'The best estimator for the XGBoost regressor is {RCV.best_estimator_}')

            # Building the best estimator recommended by the randomized search CV as the final XGBoosting regressor.
            XGBR = RCV.best_estimator_

            # Fitting on train data
            XGBR.fit(self.X_train, self.y_train)

            # Feature importance by the XGBoosting regressor
            XGBR_feature_imp = pd.DataFrame(XGBR.feature_importances_,
                                            index=self.X_train.columns,
                                            columns=['Feature_importance'])

            XGBR_feature_imp.sort_values(by='Feature_importance',
                                         ascending=False,
                                         inplace=True)

            print()
            print(f'Feature Importance by the XGBoost regressor {XGBR_feature_imp}')
            print()

            logging.info("Successfully built a model using XGBoost regressor ")

            logging.info('Exited the xgb_regressor method of the TreeModelsReg class')

            return XGBR

        except Exception as e:
            
            # logging operation
            logging.error('Exception occurred in xgb_regressor method of the TreeModelsReg class. Exception '
                          'message:' + str(e))
            logging.info('xgb_regressor method unsuccessful. Exited the xgb_regressor method of the Ensemble_models class')