"""
Script for a more detailed validation of the best regression parameters
(method + number of features) obtained by running "regressor_comparison.py".

The output contains the MAE scores and the list of selected features of each CV fold.
"""

import statistics
import numpy as np
import pandas as pd
import feat_lists_12th_grade # If working with 9th grade data, import 'feat_lists_12th_grade'
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Defining the feature set and target feature
feats = feat_lists_12th_grade.punct_feats
y_feat = "Asp_42_Average"

# Reading the dataset
df = pd.read_csv("data_12th_grade.csv", engine="python", encoding="utf-8")
df = df[feats]
df.dropna(inplace=True)

# Separating the target feature and predictive features
y = df[[y_feat]].to_numpy().ravel()
X_df = df.drop([y_feat], axis=1)
X = X_df.to_numpy()

# Defining the predetermined optimal parameters
regressor = SVR()
n_feats = 3

# Report file for saving the results
with open("validation_report.txt", "w") as output_f:

    # 10-fold cross-validation
    scores = []
    CV_features = []
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    for train, test in cv.split(X, y):
        # Standardizing the predictive features
        scaler = StandardScaler().fit(X[train])
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])
        # Feature selection
        selector = SelectKBest(f_regression, k = n_feats).fit(X_train, y[train])
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        # Training and testing the regression model
        model = regressor.fit(X_train, y[train])
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred >= 0, y_pred, 0) # Avoiding negative scores
        y_pred = np.where(y_pred <= 3, y_pred, 3) # Avoiding scores over 3
        score = mean_absolute_error(y[test], y_pred)
        scores.append(score)
        # Defining selected features
        feature_indices = selector.get_support(indices = True)
        selected_features_df = X_df.iloc[:,feature_indices]
        selected_features = list(selected_features_df.columns)
        CV_features.append(selected_features)
    # Writing the results to the report file
    output_f.write("Predicted feature: " + y_feat + "\n")
    output_f.write("Regressor: " + str(regressor) + "\n")
    output_f.write("Number of features: " + str(n_feats) + "\n")
    output_f.write("MAE averaged over 10-fold CV: " + str(statistics.mean(scores)) + "\n")
    output_f.write("SD of the MAE scores: " + str(statistics.stdev(scores)) + "\n")
    output_f.write("MAE scores of each CV fold:\n" + str(scores) + "\n")
    output_f.write("Features selected:\n" + str(CV_features))