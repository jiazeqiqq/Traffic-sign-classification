from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# x_train/x_test, y_train/y_test
X_train = np.load('./dataset/X_train.npy')
y_train = np.load('./dataset/y_train.npy')
X_test = np.load('./dataset/X_train.npy')
y_test = np.load('./dataset/y_train.npy')

# logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LR Accuracy:", accuracy)

# ensemble method(adaboost)
adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=2.0)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Adaboost Accuracy:", accuracy)