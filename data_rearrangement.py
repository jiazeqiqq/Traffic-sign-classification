import pandas as pd
import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

def rearrange_dataset(total_classes = int, annotations_file_loc = str, new_imageset_path=str, old_imageset_path=str, csv_file_name=str):
    os.mkdir(new_imageset_path)
    df = pd.read_csv(annotations_file_loc)
    category = df['category']
    file_name = df['file_name']
    width = df['width']
    height = df['height']
    x1 = df['x1']
    y1 = df['y1']
    x2 = df['x2']
    y2 = df['y2']
    file_name_wrt_category = list(zip(file_name, width, height, x1, y1, x2, y2, category))
    qualified_file_name_wrt_category = []
    for i in range(total_classes):
        count_per_category = 0
        temperory_list = []
        for bundle in file_name_wrt_category:
            if bundle[7] == i:
                temperory_list.append(bundle)
                count_per_category += 1
            else:
                pass
        if count_per_category > 190 and count_per_category < 600:
            for bundle in temperory_list:
                image = Image.open(os.path.join(old_imageset_path, str(bundle[0])))
                cropped_image = image.crop((int(bundle[3]), int(bundle[4]), int(bundle[5]), int(bundle[6])))
                cropped_image.save(os.path.join(new_imageset_path, str(bundle[0])))
            temperory_list = [(file_name, width, height, category) for file_name, width, height, _, _, _, _, category in temperory_list]
            qualified_file_name_wrt_category += temperory_list
            print(f"catgory: {i} with {count_per_category} numbers of data")
    df = pd.DataFrame(qualified_file_name_wrt_category[0:], columns=['image_name', 'width', 'height', 'category'])
    df.to_csv(csv_file_name, index=False)
    return qualified_file_name_wrt_category

def split_train_test(list_with_tuple_bundle = list, csv_train_file_name=str, csv_test_file_name=str):
    y = np.array([list(item)[-1] for item in list_with_tuple_bundle])
    X = np.array([list(item)[0] for item in list_with_tuple_bundle])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test = np.hstack((X_test.reshape(X_test.shape[0], 1), y_test.reshape(y_test.shape[0], 1)))
    train = np.hstack((X_train.reshape(X_train.shape[0], 1), y_train.reshape(y_train.shape[0], 1)))
    df_test = pd.DataFrame(test[0:], columns=['image_name', 'category'])
    df_test.to_csv(csv_test_file_name, index=False)
    df_train = pd.DataFrame(train[0:], columns=['image_name', 'category'])
    df_train.to_csv(csv_train_file_name, index=False)
    return test, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_classes", type=int, default=58)
    parser.add_argument("--annotations_loc", type=str, default='./dataset/annotations.csv')
    parser.add_argument("--new_image_loc", type=str, default='./dataset/new_imageset')
    parser.add_argument("--old_image_loc", type=str, default='./dataset/images')
    parser.add_argument("--csv_file_name", type=str, default='./dataset/new_annotations.csv')
    parser.add_argument("--train_csv_file_name", type=str, default='./dataset/train.csv')
    parser.add_argument("--test_csv_file_name", type=str, default='./dataset/test.csv')
    args = parser.parse_args()
    list_of_tuple = rearrange_dataset(total_classes = args.total_classes, annotations_file_loc = args.annotations_loc, new_imageset_path=args.new_image_loc, old_imageset_path=args.old_image_loc, csv_file_name = args.csv_file_name)
    print("Finish moving partial images into new_image_set")
    test, train = split_train_test(list_with_tuple_bundle = list_of_tuple, csv_train_file_name=args.train_csv_file_name, csv_test_file_name=args.test_csv_file_name)
    print("Finish split training and testing datasets")

    
