import os
import numpy as np
import pandas as pd
import cv2
import argparse
def average_w_h_caculation(annotations_file_loc = str, new_imageset_path=str, width_size=int, height_size=int):
    df = pd.read_csv(annotations_file_loc)
    n = len(df)
    category = (df['category'].values).reshape(n, 1)
    file_name = (df['image_name'].values).reshape(n, 1)
    width = (df['width'].values).reshape(n, 1)
    height = (df['height'].values).reshape(n, 1)
    dataset = np.hstack((file_name, width, height, category))
    for data in dataset:
        image = cv2.imread(os.path.join(new_imageset_path, data[0]))
        if data[1] > width_size and data[2] > height_size:
            image = cv2.resize(image, (width_size, height_size), interpolation=cv2.INTER_AREA)
        elif data[1] < width_size and data[2] < height_size:
            image = cv2.resize(image, (width_size, height_size), interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, (width_size, height_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(new_imageset_path, data[0]), image)
    dataset = np.delete(dataset, [1, 2], axis=1)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file_loc", type=str, default='./dataset/new_annotations.csv')
    parser.add_argument("--new_imageset_path", type=str, default='./dataset/new_imageset')
    args = parser.parse_args()
    dataset = average_w_h_caculation(annotations_file_loc = args.annotations_file_loc, new_imageset_path=args.new_imageset_path, width_size=128, height_size=128)
    print("Finish resizing all images")