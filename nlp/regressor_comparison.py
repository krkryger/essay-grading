"""
Script for cross-validating regression parameters used to predict the scores of essay rubrics.

The feature sets can be imported from "feat_lists_9th_grade.py" and "feat_lists_12th_grade.py":
- Punctuation (Asp 42): punct_feats
- Orthography and morphology (Asp 43): orth_feats
- Structuring and formatting (Asp 34): struct_feats
- Syntax (Asp 40): sent_feats
- Vocabulary (Asp 28): word_feats
"""

import statistics
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import feat_lists_12th_grade # If working with 9th grade data, import 'feat_lists_12th_grade'

# Defining the feature set and target feature
feats = feat_lists_12th_grade.punct_feats
y_feat = "Asp_42_Average"

# Lists of regressors and regressor names
regressors = [
    LinearRegression(),
    Ridge(random_state=0),
    Lasso(random_state=0),
    ElasticNet(random_state=0),
    SVR(),
    RandomForestRegressor(random_state=0),
    DecisionTreeRegressor(random_state=0)
]

reg_names = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Elastic Net",
    "Support Vector Regression",
    "Random Forest Regression",
    "Decision Tree Regression"
]

# Reading the dataset
df = pd.read_csv("data_12th_grade.csv", engine="python", encoding="utf-8")
df = df[feats]
df.dropna(inplace=True)

# Separating the target feature and predictive features
y = df[[y_feat]].to_numpy().ravel()
X_df = df.drop([y_feat], axis=1)
X = X_df.to_numpy()

# Report file for saving the results
with open("comparison_report.txt", "w") as output_f:

    # Comparing different regressors with a varied number of features
    for n_features in range(1, len(feats)):
        output_f.write("Number of features in model: " + str(n_features) + "\n\n")
        for name, reg in zip(reg_names, regressors):
            scores = []
            # 10-fold cross-validation on the whole dataset
            cv = KFold(n_splits=10, shuffle=True, random_state=0)
            for train, test in cv.split(X, y):
                # Standardizing the predictive features
                scaler = StandardScaler().fit(X[train])
                X_train = scaler.transform(X[train])
                X_test = scaler.transform(X[test])
                # Feature selection
                selector = SelectKBest(f_regression, k = n_features).fit(X_train, y[train])
                X_train = selector.transform(X_train)
                X_test = selector.transform(X_test)
                # Training and testing the regression model
                model = reg.fit(X_train, y[train])
                y_pred = model.predict(X_test)
                y_pred = np.where(y_pred >= 0, y_pred, 0) # Avoiding negative scores
                y_pred = np.where(y_pred <= 3, y_pred, 3) # Avoiding scores over 3
                score = mean_absolute_error(y[test], y_pred)
                scores.append(score)
            # Writing the results to the report file
            output_f.write(name + ":\n")
            output_f.write("MAE averaged over 10-fold CV: " + str(statistics.mean(scores)) + "\n")
            output_f.write("SD of the MAE scores: " + str(statistics.stdev(scores)) + "\n\n")
