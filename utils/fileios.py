import os
import json
import pandas as pd
import errno
import csv
import numpy as np
import shutil
from copy import deepcopy


def mkdir_if_missing(directory: str):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def dump_json(filename: str, in_data):
    if not filename.endswith('.json'):
        filename += '.json'
    data_to_write = []
    if os.path.exists(filename):
        try:
            # 1. 读取现有数据
            with open(filename, 'r') as fbj:
                existing_data = json.load(fbj)
                if isinstance(existing_data, list):
                    # 如果现有数据是列表，直接将新数据作为元素添加
                    data_to_write = existing_data
                    data_to_write.append(in_data)
                elif isinstance(existing_data, dict):
                    if "entries" in existing_data and isinstance(existing_data["entries"], list):
                        existing_data["entries"].append(in_data)
                    else:
                        existing_data["entries"] = [in_data]
                    data_to_write = existing_data
                else:
                    raise TypeError(f"Existing data has wrong data type {type(existing_data)}")
        except json.JSONDecodeError:
            # 如果文件存在但内容不是有效的JSON（例如空文件或损坏的文件）
            # 我们将其视为一个空列表
            data_to_write = [in_data]
    else:
        data_to_write = [in_data]
    with open(filename, 'w') as fbj:
        json.dump(data_to_write, fbj, indent=4)
    # with open(filename, 'w') as fbj:
    #     if isinstance(in_data, dict):
    #         json.dump(in_data, fbj, indent=4)
    #     elif isinstance(in_data, list):
    #         json.dump(in_data, fbj)
    #     else:
    #         raise TypeError(f"in_data has wrong data type {type(in_data)}")


def load_json(filename: str):
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'r') as fp:
        return json.load(fp)


def dump_txt(filename: str, in_data: str):
    if not filename.endswith('.txt'):
        filename += '.txt'

    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fbj:
        fbj.write(in_data)


def load_txt(filename: str):
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as fbj:
        return fbj.read()


def write_csv_col(filename: str, in_data):
    df = pd.DataFrame(in_data)
    df.to_csv(filename, index=False)


