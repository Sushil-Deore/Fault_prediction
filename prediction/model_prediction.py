# Importing Packages

import logging
import warnings

warnings.filterwarnings('ignore')  # ignore warnings

# configuring logging operations
logging.basicConfig(filename='Model_prediction_logs.log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


class Model_Prediction:
    """
    Description: This method makes predictions using the given model
    raises an exception if it fails

    parameters: model= model to be used for making predictions
    X = A pandas dataframe with independent features

    returns: The predictions of the target variable.
    """

    def __init__(self, model, X):
        self.model = model
        self.X = X


    def model_prediction(model, X):

        try:
            prediction = model.predict(X)

            return prediction

        except Exception as e:
            # logging operation
            logging.error('Exception occurred in "model_predict" method of the Model_Prediction class. Exception '
                          'message:' + str(e))

            logging.info('"model_predict" method unsuccessful. Exited the "model_predict" method of the '
                         'Model_Prediction class ')