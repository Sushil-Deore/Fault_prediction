
# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Configuring logging operations
logging.basicConfig(filename='Preprocessing_Development_log.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class DataPreProcessor:
    """This class is used to preprocess the data for modelling
        parameters
        dataframe: A pandas dataframe that has to be preprocessed
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def data_split(self, test_size):
        """ Description: This method splits the dataframe into train and test data respectively
            using the sklearn's "train_test_split" method.
            Raises an exception if it fails.

            Parameters: test_size: Percentage of the Dataframe to be taken as a test set

            returns: training and testing dataframes respectively.
        """

        # logging operation
        logging.info('Entered the data_split method of the DataPreProcessor class')

        try:
            df_train, df_test = train_test_split(self.dataframe, test_size=test_size, shuffle=True, random_state=42)

            # logging operation
            logging.info(
                f'Train test split successful. The shape of train data set is {df_train.shape} and the shape of '
                f'test data set is {df_test.shape}')
            logging.info('Exited the data_split method of the DataPreprocessor class ')

            return df_train, df_test

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in data_split method of the DataPreProcessor class. Exception '
                          'message:' + str(e))
            logging.info('Train test split unsuccessful. Exited the data_split method of the '
                         'DataPreProcessor class ')

    def feature_engineering(self):
        """Description: This method does the feature engineering of the features of both the train and test datasets
        respectively, using the dummy variables method.
        Raises an exception if it fails.

        parameters: columns with datatype as object

        returns: dummy variable
        """
        logging.info('Entered the feature_engineering method of the DataPreprocessor class')

        try:
            # Column sex
            sex_dummies = pd.get_dummies(self.dataframe.sex, drop_first=True)
            self.dataframe = pd.concat([self.dataframe, sex_dummies], axis=1)

            # Column children
            children_dummies = pd.get_dummies(self.dataframe.children, prefix='Children', drop_first=True)
            self.dataframe = pd.concat([self.dataframe, children_dummies], axis=1)

            # Column smoker
            smoker_dummies = pd.get_dummies(self.dataframe.smoker, prefix='Smoker', drop_first=True)
            self.dataframe = pd.concat([self.dataframe, smoker_dummies], axis=1)

            # Column region
            region_dummies = pd.get_dummies(self.dataframe.region, prefix='region', drop_first=True)
            self.dataframe = pd.concat([self.dataframe, region_dummies], axis=1)

            self.dataframe = self.dataframe.drop(['sex','children','smoker','region'], axis=1)

            return self.dataframe

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in feature_engineering method of the DataPreProcessor class. '
                          'Exception message:' + str(e))
            logging.info('feature_engineering unsuccessful. Exited the feature_engineering method of the '
                         'DataPreProcessor class ')


    def feature_scaling(self, df_train, df_test):
        """Description: This method scales the features of both the train and test datasets
        respectively, using the sklearn's "StandardScaler" method.
        Raises an exception if it fails.

        parameters: df_train: A pandas dataframe representing the training data set
        df_test: A pandas dataframe representing the testing data set

        returns: training and testing dataframes in a scaled format.
        """
        # logging operation
        logging.info('Entered the feature_scaling method of the DataPreprocessor class')

        try:
            columns = df_train.columns
            scalar = StandardScaler()
            df_train = scalar.fit_transform(df_train)
            df_test = scalar.transform(df_test)

            # logging operation

            logging.info('Feature scaling of both train and test datasets successful. Exited the feature_scaling method of the DataPreProcessor class')

            # converting the numpy arrays into pandas Dataframe
            df_train = pd.DataFrame(df_train, columns=columns)
            df_test = pd.DataFrame(df_test, columns=columns)

            return df_train, df_test

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in feature_scaling method of the DataPreProcessor class. '
                          'Exception message:' + str(e))
            logging.info('Feature scaling unsuccessful. Exited the feature_scaling method of the '
                         'DataPreProcessor class ')

    def train_test_splitting(self, df_train, df_test, column_name):
        """Description: This method splits the data into dependent and independent variables respectively
        i.e., X and y.
        Raises an exception if it fails

        parameters:
        df_train: A pandas dataframe representing the training data set
        df_test: A pandas dataframe representing the testing data set
        column_name: Target column or feature, which has to be predicted using other features

        returns:
        independent and dependent features of the both training and testing datasets respectively.
        i.e., df_train into X_train, y_train and df_test into X_test, y_test respectively.
        """

        # logging operation
        logging.info('Entered the splitting_as_X_y method of the DataPreprocessor class')

        try:
            X_train = df_train.drop(column_name, axis=1)
            y_train = df_train[column_name]
            X_test = df_test.drop(column_name, axis=1)
            y_test = df_test[column_name]

            # logging operation
            logging.info(f'Splitting data into X and y is successful. Shapes of X_train is {X_train.shape},'
                         f'y_train is {y_train.shape}, '
                         f'X_test is {X_test.shape} & '
                         f'the y_test is {y_test.shape}')
            logging.info('Exited the train_test_splitting method of DataPreProcessor class')

            return X_train, y_train, X_test, y_test

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in splitting_as_X_y method of the DataPreprocessor class. '
                          'Exception message:' + str(e))
            logging.info('Splitting data into X and y is unsuccessful. Exited the train_test_splitting method of the '
                         'DataPreProcessor class')