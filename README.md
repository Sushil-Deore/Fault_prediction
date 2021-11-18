# Insurance Premium Prediction:
 
To give people an estimate of how much they need based on their individual health situation. After that, customers can work with any health insurance carrier and its plans and perks while keeping the projected cost from our study in mind. I am considering variables as age, sex, BMI, number of children, smoking habits and living region to predict the premium. This can assist a person in concentrating on the health side of an insurance policy rather than the ineffective part.

**_ Data source _**: https://www.kaggle.com/noordeen/insurance-premium-prediction

## Approach: 
1. Loading the dataset using Pandas and performed basic checks like the data type of each column and having any missing values.
2. Performed Exploratory data analysis:
- Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
- To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
- the distribution of the target feature, expenses which was in Normal distribution with a very little right skewness.
- Checked for the presence of outliers in all the columns
3. Experimenting with various ML algorithms
- First, tried with Linear regression models, ridge and lasso regression approached. Performance metrics are calculated for all the approaches. The test RMSE score is little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression.
- Next, tried with various tree based models, performed hyper parameter tuning using the GridSearchCV and found the best hyperparameters for each model. Then, picked the top most features as per the feature importance by an each model. Models, evaluated on both the training and testing data and recorded the performance metrics.
- Based on the performance metrics of both the linear and the tree based models, XGBoost regressor performed the best, followed by the random forest regressor.
4.Deployment: Deployed the XGBoost regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML5.

At each step in both development and deployment parts, logging operation is performed which are stored in the Jupyter_Notebook_logs.log and app_deployment_logs.log files respectively

So, now we can find the insurance premium quickly by just passing the mentioned details as an input to the web application 😊

## Web Deployment
Deployed on web using Heroku (PaaS) url: https://insurancepremiumpred.herokuapp.com/

## Screenshots
![UI](https://user-images.githubusercontent.com/69260855/142414181-67630ea9-48db-4a73-92f2-624df0984341.png)

## Tools and Technologies used

![Tools](https://user-images.githubusercontent.com/69260855/142414506-f21e3ea1-5956-418e-903d-9835c32f3708.png)

## High Level Design: URL:

## Low Level Design: URL:

## Architecture: URL:

## Detailed Project report: URL:

## Wireframe document: URL:

## Demo video: URL: 

## Reference:
- https://scikitlearn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

- https://www.google.com/url?sa=D&q=https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b&ust=1637241300000000&usg=AOvVaw0pO3x0h1T83PvMf_TFBEI7&hl=en

- https://towardsdatascience.com/3-techniques-to-avoid-overfitting-of-decision-trees-1e7d3d985a09

- https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

- https://www.w3schools.com/

- https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax

## Author:

- ###### Sushil Ramdas Deore (Linkedin: https://www.linkedin.com/in/sushil-deore/)
