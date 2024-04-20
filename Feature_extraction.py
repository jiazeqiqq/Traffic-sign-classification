import extract_hog as eh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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