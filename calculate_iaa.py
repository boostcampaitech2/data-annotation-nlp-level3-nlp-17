import pandas as pd
import numpy as np
import os
import re
import json
from transformers import (
    HfArgumentParser,
)
from config import (
    IaaArguments,
    DataArguments,
)
from typing import Dict, NoReturn
from fleiss import fleissKappa

def load_json(filepath :str) ->Dict:
    with open(filepath, "r") as json_file:
        output = json.load(json_file)

    return output

if __name__ == '__main__':
    parser = HfArgumentParser(
        (IaaArguments, DataArguments)
    )
    iaa_args, data_args = parser.parse_args_into_dataclasses()

    iaa_dataset = pd.DataFrame()
    iaa_path = os.path.join(iaa_args.iaa_path, iaa_args.iaa_map_path)
    iaa_relation_map = load_json(iaa_path)

    file_list = [os.path.join(data_args.data_path, iaa_args.iaa_target, i) for i in
                 os.listdir(os.path.join(data_args.data_path, iaa_args.iaa_target)) if ".xlsx" in i]

    for file_name in file_list:
        worker = re.findall("T\d+", file_name)[0]
        output = pd.read_excel(file_name, engine='openpyxl')
        output[worker] = output["class"].apply(lambda x: iaa_relation_map[x])
        iaa_dataset[worker] = output[worker]

    if iaa_args.iaa_make_output:
        output = pd.read_excel(file_name, engine='openpyxl')
        iaa_dataset.to_excel(os.path.join(iaa_args.iaa_path, iaa_args.iaa_target+"_iaa.xlsx"), engine='openpyxl')

    iaa_dataset = iaa_dataset.to_numpy()
    num_classes = int(np.max(iaa_dataset))

    transformed_result = []
    for i in range(len(iaa_dataset)):
        temp = np.zeros(num_classes)
        for j in range(len(iaa_dataset[i])):
            temp[int(iaa_dataset[i][j] - 1)] += 1
        transformed_result.append(temp.astype(int).tolist())

    kappa = fleissKappa(transformed_result, len(iaa_dataset[0]))





