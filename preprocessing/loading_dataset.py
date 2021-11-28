# Importing packages

import pandas as pd
import logging

# configuring logging operation

logging.basicConfig(filename='development_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class Dataload:
    """
    This class is used to fetch data for the training.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def fetch_data(self):
        """
        This method reads data from source and returns a pandas dataframe
        Raises an exception if it fails.

        parameters: dataset in the .csv format

        Returns: Given dataset in the form of Pandas Dataframe
        """
        # logging operation
        logging.info('Entered the "fetch_data" method of the "DataLoad" class.')

        try:
            # reading the dataset
            df = pd.read_csv(self.dataset)

            # logging operation
            logging.info(f'Data loaded successfully. Shape of the data is {df.shape}')
            logging.info('Exited the "fetch_data" method of the "DataLoad" class.')

            return df

        except Exception as e:

            # logging operation

            logging.error('Exception occurred in fetch_data method of the DataLoad class. Exception message:' + str(e))

            logging.info('Data fetching unsuccessful.Exited the "fetch_data" method of the "DataLoad" class.')