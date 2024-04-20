import extract_hog as eh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans

# Load training images from train.csv
train = pd.read_csv('./dataset/train.csv')
image_names = train['image_name'].tolist()

path_prefix = './dataset/train_images/'
path_prefix = "./dataset/new_imageset/"
full_paths = [path_prefix + name for name in image_names]

# Extract HOG features for each image
features_list = []
for path in full_paths:
    features, _, _ = eh.extract_hog_features(path)
    features_list.append(features)

# Convert the list to a 2D array
X_train = np.array(features_list)
print("X_train:", X_train.shape)
X_label = train['category'].to_list()
X_label = np.array(X_label)
print("X_label:", X_label.shape)

# Perform LDA
lda = LDA(n_components=3)
X_r = lda.fit_transform(X_train, X_label)
print("X_r:", X_r.shape)

#perform K-mean
orginal_label = np.unique(X_label)
k = len(orginal_label)
y_pred = KMeans(n_clusters=k, random_state=42).fit_predict(X_r)

#visulization part
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'gold', 'gray', 'cyan', 'coral']
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
for i in range(k):
    ax1.scatter(X_r[(y_pred == i), 0], X_r[(y_pred == i), 1], X_r[(y_pred == i), 2], marker='o',c=colors[i], label=f'Cluster {i}')
    ax2.scatter(X_r[(X_label == orginal_label[i]), 0], X_r[(X_label == orginal_label[i]), 1], X_r[(X_label == orginal_label[i]), 2], marker='o', c=colors[i], label=f'Cluster {i}')
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