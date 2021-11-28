# Importing modules

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
import logging
import statsmodels.api as sm
from sklearn.feature_selection import RFE
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

    """
    def forward_selection(self, significance_level=0.05):

        # function accepts X_train, y_train and significance level as arguments and returns
        # the linear regression model, its predictions of both the training and testing dataframes and the relevant features.

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
    """

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

            x_train_BE = self.X_train[best_features_BE]  # considering only the relevant features
            x_test_BE = self.X_test[best_features_BE]  # considering only the relevant features

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

    def rfe_approach(self):
        """Description: This method uses Recursive Feature Elimination algorithm of sci-kit learn, which ultimately
         selects the most relevant features of the given dataset.
         Raises an exception if it fails.
        returns: returns the linear regression model, its predictions on both the training and testing dataframes and the relevant
        features.
         """
        logging.info(
            'Entered the rfe_approach method of the LR_With_FeatureSelection class')  # logging operation

        try:

            # taking all the column names into a list
            features = self.X_train.columns.tolist()

            nof_list = np.arange(1, len(features) + 1)

            # variable which stores the highest score among all the variables
            high_score = 0

            # variable to store the number of optimum features
            nof = 0

            # scores of all the variables
            score_list = []

            for n in range(len(nof_list)):
                # instantiating LinearRegression object, imported from sci-kit learn
                lr = LinearRegression()

                # instantiating RFE, imported from sci-kit learn
                rfe = RFE(lr, n_features_to_select=nof_list[n])

                # fitting RFE on the train data
                X_train_rfe = rfe.fit_transform(self.X_train, self.y_train)

                # transforming the test data using RFE
                X_test_rfe = rfe.transform(self.X_test)

                # fitting the LinearRegression model
                lr.fit(X_train_rfe, self.y_train)

                # collecting scores and appending to list
                score = lr.score(X_test_rfe, self.y_test)
                score_list.append(score)
                if score > high_score:
                    high_score = score
                    nof = nof_list[n]

            # initiating RFE with optimum features only
            lr = LinearRegression()
            rfe = RFE(lr, n_features_to_select=nof)

            # fitting RFE on the train data
            x_train_rfe = rfe.fit_transform(self.X_train, self.y_train)

            # transforming the test data using RFE
            x_test_rfe = rfe.transform(self.X_test)

            # Building Linear regression
            lr.fit(x_train_rfe, self.y_train)

            # storing rfe supported columns into a pandas series
            temp = pd.Series(rfe.support_, index=features)

            selected_features_rfe = temp[temp == True].index

            # displaying the rfe selected features
            print()
            print('Features selected by the RFE method in Linear regression are', selected_features_rfe)
            print()

            # predictions on the data
            y_pred_train_rfe = lr.predict(x_train_rfe)
            y_pred_test_rfe = lr.predict(x_test_rfe)

            # logging operation
            logging.info('Linear regression model built successfully using RFE approach. ')

            # logging operation
            logging.info('Exited the rfe_approach method of the LR_With_FeatureSelection class')

            return lr, x_train_rfe, y_pred_train_rfe, x_test_rfe, y_pred_test_rfe, selected_features_rfe

        except Exception as e:
            logging.error(
                'Exception occurred in rfe_approach method of the LR_With_FeatureSelection class. Exception '
                'message:' + str(e))  # logging operation
            logging.info('RFE method unsuccessful. Exited the rfe_approach method of the '
                         'LR_With_FeatureSelection class ')  # logging operation


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
            print('Feature importance by the ElasticNetCV are : ')
            print(imp_Coef)

            y_pred_train_ECV = ECV.predict(self.X_train)  # predictions on the train data
            y_pred_test_ECV = ECV.predict(self.X_test)  # predictions on the test data

            # logging operation

            logging.info('Linear regression model built successfully using Elastic_Net_CV approach. ')

            logging.info('Exited the Elastic_Net_CV method of the Embedded_method_for_feature_selection class.')

            return ECV, self.X_train, y_pred_train_ECV, self.X_test, y_pred_test_ECV
        except Exception as e:
            # logging operation
            logging.error(
                'Exception occurred in Elastic_Net_CV method of the Embedded_method_for_feature_selection class. Exception '
                'message:' + str(e))
            logging.info('Elastic_Net_CV method unsuccessful. Exited the Elastic_Net_CV method of the '
                         'Embedded_method_for_feature_selection class ')


class Lasso_CV:
    """This class is used to train the models using Linear regression with Lasso regularization or L1 regularization.
    parameters:
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

    def lassocv(self):
        """Description: This method uses LassoCV algorithm imported from the sci-kit learn library to build a regression model.
        It does a cross validation with various learning rates, ultimately finds the most relevant features
        and builds a model, i.e., redundant features will be eliminated.
        Raises an exception if it fails
        returns:
        returns the linear regression model,its predictions on both the training and testing dataframes with the features
        selected by the LassoCV.
        """
        logging.info('Entered the lassocv method of the Lasso class.')  # logging operation

        try:
            # Instantiating LassoCV
            ls = LassoCV()
            ls.fit(self.X_train, self.y_train)  # fitting on the training data

            coef = pd.Series(ls.coef_, index=self.X_train.columns)  # feature importance by LassoCV

            imp_coef = coef.sort_values(ascending=False)

            print('Feature importance by the LassoCV are: ', imp_coef)
            print()

            y_pred_train_lasso = ls.predict(self.X_train)  # predictions on the train data
            y_pred_test_lasso = ls.predict(self.X_test)  # predictions on the test data

            # logging operation
            logging.info('Linear regression model built successfully using LassoCV approach. ')

            logging.info('Exited the lassocv method of the Lasso class.')  # logging operation

            return ls, self.X_train, y_pred_train_lasso, self.X_test, y_pred_test_lasso

        except Exception as e:
            logging.error('Exception occurred in lassocv method of the Lasso class. Exception '
                          'message:' + str(e))  # logging operation
            logging.info('lassocv method unsuccessful. Exited the lassocv method of the '
                         'Lasso class ')  # logging operation