from data_rearrangement import rearrange_dataset, split_train_test
import argparse
# from Feature_extraction import extract_hog_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_loc", type=str, default='./dataset/annotations.csv')
    parser.add_argument("--new_image_loc", type=str, default='./dataset/new_imageset')
    parser.add_argument("--old_image_loc", type=str, default='./dataset/images')
    args = parser.parse_args()
    list_of_tuple = rearrange_dataset(annotations_file_loc = args.annotations_loc, new_imageset_path=args.new_image_loc, old_imageset_path=args.old_image_loc)
    X_train, X_test, y_train, y_test = split_train_test(list_with_tuple_bundle = list_of_tuple)
    print('x_train:', X_train)
    print('y_train:', y_train)
    print('x_test:', X_test)
    print('y_test:', y_test)
