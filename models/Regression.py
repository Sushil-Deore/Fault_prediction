# Importing modules

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, ElasticNetCV
import logging
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# configuring logging operations
logging.basicConfig(filename='Regression_development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class LR_With_FeatureSelection:
    """This class is used to build Linear regression models with only the relevant features.
        reference_1: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
        reference_2: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

        Parameters:
        X_train: Training data frame containing the independent features.
        y_train: Training dataframe containing the dependent or target feature.
        X_test: Testing dataframe containing the independent features.
        y_test: Testing dataframe containing the dependent or target feature.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def forward_selection(self, significance_level=0.05):

        """ function accepts X_train, y_train and significance level as arguments and returns
            the linear regression model, its predictions of both the training and testing dataframes and the relevant features.
        """

        # Logging operation
        logging.info('Entered the Forward selection method of the LR_With_FeatureSelection class')

        try:
            initial_features = self.X_train.columns.tolist()
            best_features_FS = []  # All the relevant columns in a list

            while len(initial_features) > 0:
                remaining_features = list(set(initial_features) - set(best_features_FS))

                new_p_value = pd.Series(index=remaining_features)

                for new_column in remaining_features:
                    model = sm.OLS(self.y_train, sm.add_constant(self.X_train[best_features_FS + [new_column]])).fit()
                    new_p_value[new_column] = model.pvalues[new_column]

                min_p_value = new_p_value.min()

                if min_p_value < significance_level:
                    best_features_FS.append(new_p_value.idxmin())
                else:
                    break
            print()
            print('Features selected by Forward selection method in Linear regression are ', best_features_FS)
            print()

            X_train_FS = self.X_train[best_features_FS]
            X_test_FS = self.X_test[best_features_FS]

            lr = LinearRegression()

            lr.fit(X_train_FS, self.y_train)

            y_pred_train_FS = lr.predict(X_train_FS)
            y_pred_test_FS = lr.predict(X_test_FS)

            # logging operation
            logging.info('Linear regression model built successfully using Forward Selection method.')

            logging.info(
                'Exited the Forward Selection method method of the LR_With_FeatureSelection class')

            return lr, X_train_FS, y_pred_train_FS, X_test_FS, y_pred_test_FS, best_features_FS

        except Exception as e:

            # logging operation

            logging.error(
                'Exception occurred in Forward selection approach method of the LR_With_FeatureSelection class. Exception message:' + str(
                    e))
            logging.info(
                'Forward selection method unsuccessful. Exited the Forward selection method of the LR_With_FeatureSelection class')


    def backward_elimination(self, Significance_level=0.05):
        """Description: This method builds a linear regression model on all the features and eliminates
        each one w.r.t. its p-value if it is above 0.05. Else it will be retained Raises an exception if it fails.

        returns the linear regression model, its predictions of both the training and testing dataframes and the relevant features.
        """

        # Logging operation
        logging.info('Entered the Backward Elimination method of the LR_With_FeatureSelection class')
        try:
            best_features_BE = self.X_train.columns.tolist()
            while len(best_features_BE) > 0:
                features_with_constant = sm.add_constant((self.X_train[best_features_BE]))
                p_values = sm.OLS(self.y_train, features_with_constant).fit().pvalues[1:]
                max_p_value = p_values.max()

                if max_p_value >= Significance_level:
                    excluded_features = p_values.idxmax()
                    best_features_BE.remove(excluded_features)
                else:
                    break

            print()
            print("Features selected by the Backward elimination method in Linear regression are ", best_features_BE)
            print()

            x_train_BE = self.x_train[best_features_BE]  # considering only the relevant features
            x_test_BE = self.x_test[best_features_BE]  # considering only the relevant features

            lr = LinearRegression()  # instantiating linear regression model from LinearRegression of sci-kit learn

            lr.fit(x_train_BE, self.y_train)  # fitting

            y_pred_train_BE = lr.predict(x_train_BE)  # predictions on train data
            y_pred_test_BE = lr.predict(x_test_BE)  # predictions on test data

            # logging operation
            logging.info('Linear regression model built successfully using Backward Elimination approach.')

            logging.info(
                'Exited the backward_elimination method of the LR_With_FeatureSelection class')

            return lr, x_train_BE, y_pred_train_BE, x_test_BE, y_pred_test_BE, best_features_BE

        except Exception as e:
            # logging operation
            logging.error(
                'Exception occurred in backward_elimination method of the LR_With_FeatureSelection class. Exception '
                'message:' + str(e))
            logging.info(
                'Backward elimination method unsuccessful. Exited the backward_elimination method of the '
                'LR_With_FeatureSelection class')


class Embedded_method_for_feature_selection:
    """This class is used to train the models using Linear regression with Elastic Net model with iterative fitting along a regularization path.
        parameters
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

    def Elastic_Net_CV(self):
        # logging operation
        logging.info('Entered the Elastic_Net_CV method of the Embedded_method_for_feature_selection class.')

        try:
            ECV = ElasticNetCV()  # Instantiating ElasticNetCV
            ECV.fit(self.X_train, self.y_train)  # fitting on the training data

            Coef = pd.Series(ECV.coef_, index=self.X_train.columns)  # feature importance by ElasticNetCV

            imp_Coef = Coef.sort_values(ascending=False)

            print()
            print(f'Feature importance by the ElasticNetCV are : {imp_Coef}')
            print()

            y_pred_train_ECV = ECV.predict(self.X_train)  # predictions on the train data
            y_pred_test_ECV = ECV.predict(self.X_test)  # predictions on the test data

            # logging operation

            logging.info('Linear regression model built successfully using Elastic_Net_CV approach. ')

            logging.info(
                'Exited the Elastic_Net_CV method of the Embedded_method_for_feature_selection class.')  # logging operation

            return ECV, self.X_train, y_pred_train_ECV, self.X_test, y_pred_test_ECV
        except Exception as e:
            # logging operation
            logging.error('Exception occurred in Elastic_Net_CV method of the Embedded_method_for_feature_selection class. Exception '
                          'message:' + str(e))
            logging.info('Elastic_Net_CV method unsuccessful. Exited the Elastic_Net_CV method of the '
                         'Embedded_method_for_feature_selection class ')
