import extract_hog as eh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

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

# Autoencoder
feature_size = 8100  # Size of the HOG feature vector
latent_dim = 128  # Size of the latent space

# Input layer
input_layer = Input(shape=(feature_size,))

# Encoder
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(latent_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(feature_size, activation='sigmoid')(decoded)

# Autoencoder Model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

