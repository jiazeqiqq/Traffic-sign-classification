from data_rearrangement import rearrange_dataset, split_train_test
import argparse
import numpy as np

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
    test, train = split_train_test(list_with_tuple_bundle = list_of_tuple, csv_train_file_name=args.train_csv_file_name, csv_test_file_name=args.test_csv_file_name)
    print("test:", test)
    print("test_shape:", test.shape)
    print("train:", train)
    print("train_shape:", train.shape)
    print("orginal_list:", np.array(list_of_tuple))
    print("orginal_list_shape:", np.array(list_of_tuple).shape)
