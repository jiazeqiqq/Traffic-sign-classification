from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import pearsonr

# load x_train/x_test, y_train/y_test
X_train = np.load('./dataset/X_train.npy')
y_train = np.load('./dataset/y_train.npy')
X_test = np.load('./dataset/X_train.npy')
y_test = np.load('./dataset/y_train.npy')

# logistic regression
logreg = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LR Accuracy:", accuracy)
print("")

# ensemble method(adaboost)
adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.3)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Adaboost Accuracy:", accuracy)
print("")

# Naive Bayes(assume features follows gaussian distribution)
print("Pearson Correlation:")
pearson_corr = np.corrcoef(X_train, rowvar=False)
print(pearson_corr)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)
