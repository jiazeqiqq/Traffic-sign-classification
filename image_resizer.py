import os
import numpy as np
import pandas as pd
import cv2
def average_w_h_caculation(annotations_file_loc = str, new_imageset_path=str):
    df = pd.read_csv(annotations_file_loc)
    n = len(df)
    category = (df['category'].values).reshape(n, 1)
    file_name = (df['image_name'].values).reshape(n, 1)
    width = (df['width'].values).reshape(n, 1)
    height = (df['height'].values).reshape(n, 1)
    dataset = np.hstack((file_name, width, height, category))
    for data in dataset:
        image = cv2.imread(os.path.join(new_imageset_path, data[0]))
        if data[1] > 128 and data[2] > 128:
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        elif data[1] < 128 and data[2] < 128:
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(new_imageset_path, data[0]), image)
    dataset = np.delete(dataset, [1, 2], axis=1)
    return dataset

if __name__ == "__main__":
    dataset = average_w_h_caculation(annotations_file_loc = './dataset/new_annotations.csv', new_imageset_path='./dataset/new_imageset')
    print(dataset)