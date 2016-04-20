import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn import preprocessing

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

X = preprocessing.scale(train)
Y = preprocessing.scale(test)
features = list(train.columns[1:-5])
label = list(train["signal"])
print("Preprocessing data")

clf = svm.SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,gamma=0.001, kernel="linear", max_iter=-1, probability=False,random_state=None, shrinking=True, tol=0.001, verbose=False)
clf.fit(train[features], label)

def get_score(clf, train_features, train_labels):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_features, train_labels, test_size=0.4, random_state=0)
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test) 

print("Training Support Vector Machine")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

num_trees=250
print("Make predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("rf_xgboost_submission.csv", index=False)