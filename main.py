import logging
import warnings

import joblib
import pandas as pd

from Evaluation.evaluation import Metrics
from models.Ensemble_models import Ensemble_models
from models.Regression import LR_With_FeatureSelection, Embedded_method_for_feature_selection, Lasso_CV
from models.tree_models import Tree_models_regression
from preprocessing.dataset_loading import Dataload
from preprocessing.preprocessing import DataPreProcessor
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 10)

# Source of the dataset
data = r'dataset\insurance.csv '

# configuring logging operations
logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')

# logging operation
logging.info("The entire development process runs in the main.py ")

# An object which is responsible for the getting data from the dataset
d = Dataload(data)

# Output is nothing but the data in the form of pandas dataframe
df = d.fetch_data()

# An object which is responsible for data preprocessing
dp = DataPreProcessor(df)

# Feature engineering on columns having object as data type
df = dp.feature_engineering()

# splitting the dataframe into train and test sets respectively.
df_train, df_test = dp.data_split(test_size=0.30)

# feature scaling using StandardScaler for linear regression
df_train_sc, df_test_sc = dp.feature_scaling(df_train, df_test)

# Splitting data into independent and dependent variables respectively

X_train, y_train, X_test, y_test = dp.train_test_splitting(df_train_sc, df_test_sc, 'expenses')

# An object which is responsible for building Linear regression models
lr = LR_With_FeatureSelection(X_train, y_train, X_test, y_test)

""" Linear regression with Backward elimination approach """
# Linear regression model, its predictions using only the relevant features chose by backward elimination approach.
lr_be, X_train_be, y_pred_train_be, X_test_be, y_pred_test_be, relevant_features_by_BE = lr.backward_elimination()

logging.info("Defining a function to record the relevant features of each model in the 'relevant_features_by_models.csv' file in the 'results' directory")

# Blank dataframe to record the relevant features w.r.t. the algorithm used
imp_f = pd.DataFrame(columns=['Algorithm', 'Imp_Features'])


def rec_imp_features(dataframe, algorithm_name, imp_features):
    """Description: This function stores the important features of an algorithm we used, in a Pandas dataframe.
     parameters: dataframe: A Pandas dataframe in which the results are recorded
     imp_features: Important features chose by the Algorithm
     returns: Pandas dataframe containing the important features information w.r.t. the algorithm, saved to local disk as a .csv file.
     """
    try:
        dataframe.loc[len(dataframe)] = [algorithm_name, imp_features]

        logging.info(f"Important features of the {algorithm_name} updated in the 'relevant_features_by_models.csv' ")

        return dataframe.to_csv('results/relevant_features_by_models.csv', index=False)

    except Exception as e:
        logging.error("Exception occurred in the function 'rec_imp_features'. "
                      "Exception message: " + str(e))


# calling the function to store the info of relevant features
rec_imp_features(imp_f, "Linear_Regression_BE", relevant_features_by_BE)

# instantiating the Metrics object
metrics = Metrics()

# Data frame to record performance of every model
algo_results = pd.DataFrame(columns=['Algorithm',
                                     'Train_R2 score',
                                     'Train_Adj_R2 score',
                                     'Train_RMSE score',
                                     'Test_R2 score',
                                     'Test_Adj_R2 score',
                                     'Test_RMSE score'])

# logging operation
logging.info('Created a blank data frame "algo_results" to store the results of each and every model')

logging.info('Defining a function "evaluate" to evaluate the performance metrics of a model')


# Function to update the dataframe 'algo_results'
# noinspection PyShadowingNames
def evaluate(dataframe, algorithm_name, X_train, y_train, y_pred_train, X_test, y_test, y_pred_test):
    """ Description: This function stores the model's performance metrics and returns a pandas dataframe.
    parameters:
    dataframe: A pandas dataframe to store the results of experiments on various algorithms
    algorithm_name: Name of an algorithm w.r.t. its results
    X_train: Training data with independent features
    y_train: Training data with dependent feature i.e., actual values
    y_pred_train: Predictions on the training data
    X_test: Testing data with independent features
    y_test: Testing data with dependent feature i.e, actual values
    y_pred_test: Predictions on the testing data

    Returns: Pandas dataframe containing the performance metrics of algorithms and saved to local disk as .csv file
    """
    try:
        dataframe.loc[len(dataframe)] = [algorithm_name, metrics.r2_score(y_train, y_pred_train),
                                         metrics.adj_r2_score(X_train, y_train, y_pred_train),
                                         metrics.rmse_score(y_train, y_pred_train),
                                         metrics.r2_score(y_test, y_pred_test),
                                         metrics.adj_r2_score(X_test, y_test, y_pred_test),
                                         metrics.rmse_score(y_test, y_pred_test)]
        logging.info(f'Results Dataframe saved to disk as "Performance of algorithms.csv" file in the "results" directory. ')
        logging.info("Updated the results in 'Performance of algorithms.csv'")

        return dataframe.to_csv('results/Performance of algorithms.csv', index=False)

    except Exception as e:
        logging.error("Exception occurred in the function 'evaluate'. "
                      "Exception message: " + str(e))


# calling the function to store results into local disk
evaluate(algo_results, 'Linear Regression_BE', X_train_be, y_train, y_pred_train_be, X_test_be, y_test, y_pred_test_be)

""" 2) Linear regression with RFE approach """
# Linear regression model, its predictions using the relevant features by the RFE method.
lr_rfe, X_train_rfe, y_pred_train_rfe, X_test_rfe, y_pred_test_rfe, relevant_features_by_RFE = lr.rfe_approach()

# storing the info of the relevant features
rec_imp_features(imp_f, "Linear Regression_RFE", relevant_features_by_RFE)

# calling the function "evaluate" to store the results into .csv file
evaluate(algo_results, 'Linear Regression_RFE', X_train_rfe, y_train, y_pred_train_rfe, X_test_rfe, y_test, y_pred_test_rfe)

""" 3) Linear regression with Embedded_method_for_feature_selection approach """
# An object which is responsible for building Linear regression models with ElasticNetCV.
ECV = Embedded_method_for_feature_selection(X_train, y_train, X_test, y_test)

# Linear regression model and predictions of both the train and test data respectively
lr_ECV, X_train_ECV, y_pred_train_ECV, X_test_ECV, y_pred_test_ECV = ECV.Elastic_Net_CV()

# relevant features
relevant_features_by_ECV = ['age',
                            'yes',
                            'children',
                            'bmi',
                            'region_northwest',
                            'male',
                            'region_southwest',
                            'region_southeast']

# storing the info of the relevant features
rec_imp_features(imp_f, "Linear Regression_ECV", relevant_features_by_ECV)

# calling the function "evaluate" to store the results into local disk
evaluate(algo_results, 'Linear Regression_ECV', X_train_ECV, y_train, y_pred_train_ECV, X_test_ECV, y_test, y_pred_test_ECV)

""" 3) Linear regression with LassoCV approach """
# An object which is responsible for building Linear regression models with Lasso regularization.
ls = Lasso_CV(X_train, y_train, X_test, y_test)

# Linear regression model and predictions of both the train and test data respectively
lr_lasso, X_train_lasso, y_pred_train_lasso, X_test_lasso, y_pred_test_lasso = ls.lassocv()

# relevant features
relevant_features_by_Lasso = ['age',
                              'yes',
                              'children',
                              'bmi',
                              'region_northwest',
                              'region_southwest',
                              'male',
                              'region_southeast']

# storing the info of the relevant features
rec_imp_features(imp_f, "Linear Regression_Lasso", relevant_features_by_Lasso)

# calling the function "evaluate" to store the results into local disk
evaluate(algo_results, 'Linear Regression_Lasso', X_train_lasso, y_train, y_pred_train_lasso, X_test_lasso, y_test, y_pred_test_lasso)

"""Splitting datasets into independent and dependent datasets respectively"""

# Tree models do not require feature scaling, hence we are using the original data
X_train, y_train, X_test, y_test = dp.train_test_splitting(df_train, df_test, 'expenses')

# An object which is responsible for building tree based models
decision_tree = Tree_models_regression(X_train, y_train, X_test, y_test)

""" 1) Decision Tree Regressor """

# logging operation
logging.info("Building a Decision tree regressor model on the training data and the importance of each feature will be displayed in the console.")

# Decision tree regressor model
dt_model = decision_tree.decision_tree_regressor()

# top features as per the feature importance
top_features_dt = ['age', 'yes', 'bmi', 'children']

# storing the top features data in a file
rec_imp_features(imp_f, "Decision tree regressor", top_features_dt)

# considering only the relevant features
X_train_dt = X_train[top_features_dt]
X_test_dt = X_test[top_features_dt]

# An object which is responsible for building tree based models
decision_tree = Tree_models_regression(X_train_dt, y_train, X_test_dt, y_test)

# logging operation
logging.info("Building a Decision tree regressor model on the training data with the relevant features only")

# building a decision tree regressor model
dt_model_2 = decision_tree.decision_tree_regressor()

# predictions on the training data
y_pred_train_dt = decision_tree.model_prediction(dt_model_2, X_train_dt)

# predictions on the testing data
y_pred_test_dt = decision_tree.model_prediction(dt_model_2, X_test_dt)

logging.info('Using Decision tree regressor model, successfully made predictions on both the training and testing data respectively')

# calling the function "evaluate" to store the results into local disk.
evaluate(algo_results, 'Decision tree regressor', X_train_dt, y_train, y_pred_train_dt, X_test_dt, y_test, y_pred_test_dt)

""" 2) Random forest regressor """
# An object which is responsible for building tree based models
RF = Tree_models_regression(X_train, y_train, X_test, y_test)

# logging operation
logging.info("Building a Random forest regressor model on the training data and the importance of each feature will be displayed in the console.")

# building a random forest regressor model
rf_model = RF.random_forest_regressor()

# top 5 features as per the feature importance
top_features_rf = ['age', 'yes', 'bmi', 'children', 'male']

# storing the relevant features in a file.
rec_imp_features(imp_f, "Random Forest regressor", top_features_rf)

# considering only the relevant features
X_train_rf = X_train[top_features_rf]
X_test_rf = X_test[top_features_rf]

# An object which is responsible for building tree based models
RF = Tree_models_regression(X_train_rf, y_train, X_test_rf, y_test)

# logging operation
logging.info("Building a Random forest regressor model on the training data with the relevant features only.")

# building a random forest regressor model
rf_model_2 = RF.random_forest_regressor()

# predictions on the training data
y_pred_train_rf = RF.model_prediction(rf_model_2, X_train_rf)

# predictions on the testing data
y_pred_test_rf = RF.model_prediction(rf_model_2, X_test_rf)

logging.info('Using Random Forest Regressor model, successfully made predictions on both the training and testing data respectively')

# calling the function "evaluate" to store the results into local disk
evaluate(algo_results, 'Random Forest regressor', X_train_rf, y_train, y_pred_train_rf, X_test_rf, y_test, y_pred_test_rf)

""" 3) AdaBoost Regressor """
# An object which is responsible for building tree based models
ADB = Ensemble_models(X_train, y_train, X_test, y_test)

# logging operation
logging.info("Building an Adaboost regressor model on the training data and the importance of each feature will be displayed in the console.")

# building an Adaboost regressor model
adb_model = ADB.adaboost_regressor()

# Top features by the Adaboost regressor
top_features_adb = ['age', 'yes', 'bmi', 'children']

# storing the relevant features in the file.
rec_imp_features(imp_f, "Adaboost regressor", top_features_adb)

# considering only the relevant features
X_train_adb = X_train[top_features_adb]
X_test_adb = X_test[top_features_adb]

# An object which is responsible for building tree based models)
ADB = Ensemble_models(X_train_adb, y_train, X_test_adb, y_test)

# logging operation
logging.info("Building an Adaboost regressor model on the training data with the relevant features only.")

# building an Adaboost regressor model
adb_model_2 = ADB.adaboost_regressor()

# predictions on the training data
y_pred_train_adb = ADB.model_prediction(adb_model_2, X_train_adb)

# predictions on the testing data
y_pred_test_adb = ADB.model_prediction(adb_model_2, X_test_adb)

# logging operation
logging.info('Using Adaboost Regressor model, successfully made predictions on both the training and testing data respectively')

# calling the function "evaluate" to store the results into local disk.
evaluate(algo_results, 'Adaboost regressor', X_train_adb, y_train, y_pred_train_adb, X_test_adb, y_test, y_pred_test_adb)

""" Result:- Model accuracy is less that the random forest regressor. Let's experiment with Gradient boosting regressor."""

""" 4) Gradient Boosting Regressor """
# An object which is responsible for building tree based models
GBR = Ensemble_models(X_train, y_train, X_test, y_test)

# logging operation
logging.info("Building a Gradient Boosting regressor model on the training data and the importance of each feature will be displayed in the console.")

# building a gradient boosting regressor
gbr_model = GBR.gradientboosting_regressor()

# top 6 features by the Gradient boosting regressor
top_features_gbr = ['yes',
                    'age',
                    'bmi',
                    'children',
                    'male',
                    'region_southwest']

# storing the relevant features in the file.
rec_imp_features(imp_f, "Gradient Boost regressor", top_features_gbr)

# considering only the relevant features
X_train_gbr = X_train[top_features_gbr]
X_test_gbr = X_test[top_features_gbr]

# An object which is responsible for building tree based models
GBR = Ensemble_models(X_train_gbr, y_train, X_test_gbr, y_test)

# logging operation
logging.info("Building a Gradient Boosting regressor model on the training data with the relevant features only")

# building a Gradient boosting regressor model
gbr_model_2 = GBR.gradientboosting_regressor()

# predictions on the training data
y_pred_train_gbr = GBR.model_prediction(gbr_model_2, X_train_gbr)

# predictions on the testing data
y_pred_test_gbr = GBR.model_prediction(gbr_model_2, X_test_gbr)

logging.info('Using Gradient Boosting Regressor model, successfully made predictions on both the training and testing data respectively')

# calling the function "evaluate" to store the results into local disk
evaluate(algo_results, 'Gradient Boost regressor', X_train_gbr, y_train, y_pred_train_gbr, X_test_gbr, y_test, y_pred_test_gbr)

""" Result:- Compared to Adaboost regressor, Gradient Boost Regressor has performed well. Let's check with XGBoost regressor as well."""

""" 5) XGBoost Regressor """
# An object which is responsible for building tree based models
XGB = Ensemble_models(X_train, y_train, X_test, y_test)

# logging operation
logging.info("Building an XGBoost regressor model on the training data and the importance of each feature will be displayed in the console.")

# building an XGBoost regressor model
xgbr_model = XGB.xgb_regressor()

# top features by the XGBoost regressor
top_features_xgbr = ['age',
                     'yes',
                     'bmi',
                     'children',
                     'male',
                     'region_southwest']

# storing the relevant features into file.
rec_imp_features(imp_f, "XGBoost regressor", top_features_xgbr)

# considering only the relevant features
X_train_xgbr = X_train[top_features_xgbr]
X_test_xgbr = X_test[top_features_xgbr]

# An object which is responsible for building tree based models)
XGB = Ensemble_models(X_train_xgbr, y_train, X_test_xgbr, y_test)

# logging operation
logging.info("Building an XGBoost regressor model on the training data with the relevant features only. ")

# building a XGBoost regressor model
xgbr_model_2 = XGB.xgb_regressor()

# predictions on the training data
y_pred_train_xgbr = XGB.model_prediction(xgbr_model_2, X_train_xgbr)

# predictions on the testing data
y_pred_test_xgbr = XGB.model_prediction(xgbr_model_2, X_test_xgbr)

logging.info('Using XGBoost regressor model, successfully made predictions on both the training and testing data respectively')

# calling the function "evaluate" to store the results into local disk.
evaluate(algo_results, 'XGBoost regressor', X_train_xgbr, y_train, y_pred_train_xgbr, X_test_xgbr, y_test, y_pred_test_xgbr)

""" Result:- As per the results recorded in the "Experiments with algorithms.csv" , 
XGBoost Regressor is the best one in terms of adjusted R2 score on testing data, followed by Random Forest regressor """

# logging operation
logging.info("Best Model: XGBoost regressor ")

# logging operation
logging.info('Saving the XGBoost regressor model into the "models" directory')

# saving the best models into disk
joblib.dump(xgbr_model_2, 'pickle_files\XGBoost_Regressor_model.pkl')

# logging operation
logging.info('Saving the Random Forest regressor model into the "models" directory')

joblib.dump(rf_model_2, 'pickle_files\RandomForest_Regressor_model.pkl')

logging.info("Solution development part completed successfully. Thank you _/\_ , Stay safe and healthy :-) ")