import pandas as pd
import os
import shutil

def rearrange_dataset(annotations_file_loc = str, new_imageset_path=str, old_imageset_path=str):
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
    for i in range(58):
        count_per_category = 0
        temperory_list = []
        for bundle in file_name_wrt_category:
            if bundle[7] == i:
                temperory_list.append(bundle)
                count_per_category += 1
            else:
                pass
        if count_per_category > 190 and count_per_category < 600:
            [shutil.copy(os.path.join(old_imageset_path, str(bundle[0])), new_imageset_path) for bundle in temperory_list]
            qualified_file_name_wrt_category += temperory_list
            print("catgory: {i} with ""{count_per_category}"" numbers of data")
    return qualified_file_name_wrt_category


    
