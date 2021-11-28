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


    def removing_outliers(self, column_name):

        """ Description: This method removes outliers from the specified column using Inter quartile range method.
            Here, first we consider the values which are at the upper and lower limits and store it in one dataframe say data_included.
            Then, we exclude the values which are at the upper and lower limits and store it in one dataframe say data_excluded.
            Then, we concatenate both the data frames into a single dataframe.
            Raises an exception if it fails.

            parameters
            column_name: Column for which the outliers has to be removed.

            returns
            returns a dataframe having outliers removed in the given column.
        """

        # logging operation

        logging.info("Entered removing_outliers method of the DataPreProcessor class.")

        try:
            q1 = self.dataframe[column_name].quantile(0.25)
            q3 = self.dataframe[column_name].quantile(0.75)

            iqr = q3 - q1

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q1 + 1.5 * iqr

            data_included = self.dataframe.loc[(self.dataframe[column_name] >= lower_limit) & (self.dataframe[column_name] <= upper_limit)]

            data_excluded = self.dataframe.loc[(self.dataframe[column_name] > lower_limit) & (self.dataframe[column_name] < upper_limit)]

            self.dataframe = pd.concat([data_included, data_excluded])

            # logging operation

            logging.info(f'Outlier treatment using IQR method: Successfully removed outliers in the {column_name} column.')

            logging.info('Exited the rem_outliers method of the DataPreprocessor class ')

            return self.dataframe

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in rem_outliers method of the DataPreProcessor class. Exception '
                          'message:' + str(e))
            logging.info('Removing outliers unsuccessful. Exited the rem_outliers method of the '
                         'DataPreprocessor class ')


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


    def train_test_splitting(df_train, df_test, column_name):
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