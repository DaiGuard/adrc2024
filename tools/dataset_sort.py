import os
import pathlib
import argparse
import shutil
from xy_dataset import XYDataset



def foldersort(outputpath, prefix, minnum, dataset):

    new_dataset = []
    for i, data in enumerate(dataset):
        d = {
            "name": prefix + "/" + data["name"],
            "throttle": data["throttle"],
            "steering": data["steering"],
            "use": 0,
            "file_path": os.path.join(output_path, "images", prefix, data["name"] + "_front.jpg")
        }
        if i < minnum:
            d["use"] = 1

        new_dataset.append(d)

        shutil.copy(data["file_path"], d["file_path"])
        
    return new_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="dataset analysis")
    parser.add_argument("--data", "-d", action="append", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    args = parser.parse_args()

    # データセットの読込み
    dataset = XYDataset(args.data)

    # フォルダとファイルの準備
    output_path = pathlib.Path(args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    right_path = os.path.join(output_path, "images", "right")
    if not os.path.exists(right_path):
        os.makedirs(right_path)

    center_path = os.path.join(output_path, "images", "center")
    if not os.path.exists(center_path):
        os.makedirs(center_path)
    
    left_path = os.path.join(output_path, "images", "left")
    if not os.path.exists(left_path):
        os.makedirs(left_path)

    dataset_path = os.path.join(output_path, "dataset.csv")
    dataset_file = open(dataset_path, "w")

    # 画像ファイル読込み
    right_dataset = []
    center_dataset = []
    left_dataset = []
    for image, xy, name, file_path in dataset:
        data = {
                "name": name,
                "throttle": xy[0],
                "steering": xy[1],
                "use": 0,
                "file_path": file_path,
        }

        if xy[1] < -0.3:
            right_dataset.append(data)
        elif xy[1] > 0.3:
            left_dataset.append(data)
        else:
            center_dataset.append(data)

    min_num = min([len(left_dataset), len(center_dataset), len(right_dataset)])
    print("L: {}, C: {}, R: {}".format(len(left_dataset), len(center_dataset), len(right_dataset)))
    print("Min Count: {}".format(min_num))

    new_dataset = []
    new_dataset += foldersort(output_path, "right", min_num, right_dataset)
    new_dataset += foldersort(output_path, "center", min_num, center_dataset)
    new_dataset += foldersort(output_path, "left", min_num, left_dataset)

    for data in new_dataset:
        dataset_file.write("{}, {}, {}, {}\n".format(
            data["name"],
            data["throttle"],
            data["steering"],
            data["use"],
        ))
