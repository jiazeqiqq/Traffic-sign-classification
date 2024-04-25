import extract_hog as eh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training images from train.csv
train = pd.read_csv('./dataset/train.csv')
image_names = train['image_name'].tolist()

# Load testing images from test.csv
test = pd.read_csv('./dataset/test.csv')
test_image_names = test['image_name'].tolist()

# path_prefix = './dataset/train_images/'
path_prefix = "./dataset/new_imageset/"
full_paths = [path_prefix + name for name in image_names]
full_paths_test = [path_prefix + name for name in test_image_names]

# Extract HOG features for each image
test_feature_list = []
features_list = []
for path in full_paths:
    features, _, _ = eh.extract_hog_features(path)
    features_list.append(features)
for path in full_paths_test:
    features, _, _ = eh.extract_hog_features(path)
    test_feature_list.append(features)

# Convert the list to a 2D array
X_train = np.array(features_list)
X_test = np.array(test_feature_list)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
y_train = train['category'].to_list()
y_train = np.array(y_train)
y_test = test['category'].to_list()
y_test = np.array(y_test)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Perform LDA
lda = LDA(n_components=3)
X_train_r = lda.fit_transform(X_train, y_train)
X_test_r = lda.transform(X_test)
print("X_train_r:", X_train_r.shape)
print("X_test_r:", X_test_r.shape)

#perform K-mean
orginal_label = np.unique(y_train)
k = len(orginal_label)
y_train_pred = KMeans(n_clusters=k, random_state=42).fit_predict(X_train_r)

#visulization part
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gold', 'gray', 'cyan', 'coral']
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
for i in range(k):
    ax1.scatter(X_train_r[(y_train_pred == i), 0], X_train_r[(y_train_pred == i), 1], X_train_r[(y_train_pred == i), 2], marker='o',c=colors[i], label=f'Cluster {i}')
    ax2.scatter(X_train_r[(y_train == orginal_label[i]), 0], X_train_r[(y_train == orginal_label[i]), 1], X_train_r[(y_train == orginal_label[i]), 2], marker='o', c=colors[i], label=f'Cluster {i}')
ax1.set_xlabel('X[0]-axis')
ax1.set_ylabel('X[1]-axis')
ax1.set_zlabel('X[2]-axis')
ax1.set_title('3D Scatter Plot of data with Colored predicted Labels')
ax1.legend()
ax2.set_xlabel('X[0]-axis')
ax2.set_ylabel('X[1]-axis')
ax2.set_zlabel('X[2]-axis')
ax2.set_title('3D Scatter Plot of data with Colored  Colored true Labels')
ax2.legend()
plt.tight_layout()
plt.show()

# logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("LR Accuracy:", accuracy)

# ensemble method(adaboost)
base_estimator = RandomForestClassifier(random_state=42)
adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=200)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Adaboost Accuracy:", accuracy)