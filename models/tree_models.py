# Importing packages

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging
import warnings

warnings.filterwarnings('ignore')

# Configuring logging operations
logging.basicConfig(filename='Tree_model_development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class Tree_models_regression:
    """ This class is used to build regression models using different tree techniques.
        References:
        reference 1 - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
        reference 2 - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor
        reference 3 - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest%20regressor#sklearn.ensemble.RandomForestRegressor

        Parameters:
        x_train: Training data frame containing the independent features.
        y_train: Training dataframe containing the dependent or target feature.
        x_test: Testing dataframe containing the independent features.
        y_test: Testing dataframe containing the dependent or target feature.

        Returns:
        returns Tree models
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    """@property decorator is a built-in decorator in Python which is helpful in 
    defining the properties effortlessly without manually calling the inbuilt 
    function property(). Which is used to return the property attributes of 
    a class from the stated getter, setter and deleter as parameters
    """

    @property
    def decision_tree_regressor(self):
        """Description: This method builds a model using DecisionTreeRegressor algorithm imported from the sci-kit learn,
        by implementing cross validation technique to choose the best estimator with the best hyper parameters.
        Raises an exception if it fails

        returns: The Decision tree Regressor model and prints the importance of each feature
        """
        # Logging operation
        logging.info('Entered the Decision Tree Regressor method in class Tree_models_regression')

        try:
            # instantiating DecisionTreeRegressor object
            dt = DecisionTreeRegressor()

            # parameter grid
            params = {'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                      'max_depth': [2, 5, 8, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_samples_leaf': [2, 4, 6, 8, 10]}

            # Randomized search cross validation object imported from sci-kit learn
            RCV = RandomizedSearchCV(estimator=dt,
                                     param_distributions=params,
                                     n_iter=10,
                                     scoring='r2',
                                     cv=10,
                                     verbose=2,
                                     random_state=42,
                                     n_jobs=-1,
                                     return_train_score=True)

            print('Cross Validation process for Decision Tree Regressor')

            RCV.fit(self.X_train, self.y_train)

            print()
            print(f'The best estimator for Decision Tree Regressor is {RCV.best_estimator_} ')

            # Building the best estimator recommended by the randomized search CV as the final decision tree regressor
            dt = RCV.best_estimator_

            # fitting on the train data
            dt.fit(self.X_train, self.y_train)

            # Feature importance by the Decision Tree regressor

            df_feature_imp = pd.DataFrame(dt.feature_importances_,
                                          index=self.X_train.columns,
                                          columns=['Feature_importance'])

            df_feature_imp.sort_values(by='Feature_importance',
                                       ascending=False,
                                       inplace=True)

            print()
            print(f'Feature importance by the Decision Tree Regressor {df_feature_imp}')
            print()

            logging.info('Successfully built a model using Decision tree regressor with the best hyper parameters')
            logging.info('Exited decision_tree_regressor method of the Tree_models_regression class')

            return dt

        except Exception as e:
            # logging operation
            logging.error(
                'Exception occurred in decision_tree_regressor method of the Tree_models_regression class. Exception message:' + str(
                    e))
            logging.info(
                'decision_tree_regressor method unsuccessful. Exited the decision_tree_regressor method of the Tree_models_regression class ')

    @property
    def random_forest_regressor(self):
        """Description: This method builds a model using RandomForestRegressor algorithm, a type of ensemble technique
        imported from sci-kit learn library. It uses cross validation technique and chooses the best estimator with the
        best hyper parameters.
        Raises an exception if it fails

        returns: The Random forest regressor model and prints the importance of each feature
        """

        # logging operation
        logging.info('Entered the random_forest_regressor method in class Tree_models_regression')

        try:
            # instantiating the RandomForestRegressor object
            RF = RandomForestRegressor()

            # parameter grid
            params = {'n_estimators': [5, 10, 20, 40, 80, 100, 200],
                      'criterion': ['mse', 'mae'],
                      'max_depth': [2, 5, 10, 20],
                      'min_samples_split': [2, 4, 8, 12],
                      'min_sample_leaf': [2, 4, 6, 8, 10],
                      'oob_score': [True]}

            # instantiating RandomizedSearchCV
            RCV = RandomizedSearchCV(estimator=RF,
                                     param_distributions=params,
                                     n_iter=10,
                                     scoring='r2',
                                     cv=10,
                                     verbose=5,
                                     random_state=42,
                                     n_jobs=-1,
                                     return_train_score=True)

            print('Cross validation process for Random forest regressor')

            # Fitting on the train data
            RCV.fit(self.X_train, self.y_train)

            # displaying the best estimator
            print()
            print(f'The best estimator for the Random forest regressor is {RCV.best_estimator_}')

            # Building the best estimator recommended by the randomized search CV as the final random forest regressor.
            RF = RCV.best_estimator_

            # fitting on the train data
            RF.fit(self.X_train, self.y_train)

            # Feature importance by the Random Forest regressor

            RF_feature_imp = pd.DataFrame(RF.feature_importance_,
                                          index=self.X_train.columns,
                                          columns=['Feature_importance'])

            RF_feature_imp.sort_values(by='Feature_importance', ascending=False, inplace=True)

            print()
            print(f'Feature importance by the the Random Forest Regressor: {RF_feature_imp}')
            print()

            # logging operation
            logging.info('Successfully built a model using Random forest Regressor')
            logging.info('Exited the random_forest_regressor method of the Tree_models_regression class')

            return RF

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in random_forest_regressor method of the Tree_models_regression class. Exception '
                          'message:' + str(e))

            logging.info(
                'random_forest_regressor method unsuccessful. Exited the random_forest_regressor method of the '
                'Tree_models_regression class')
