import collections
import numpy as np
from tqdm import tqdm
import os
import random
import torch
from SPAIC import spaic
import torch.nn.functional as F
from SPAIC.spaic.Learning.Learner import Learner
from SPAIC.spaic.Library.Network_saver import network_save
from SPAIC.spaic.Library.Network_loader import network_load
from SPAIC.spaic.IO.Dataset import MNIST as dataset
# from SPAIC.spaic.IO.Dataset import CUSTOM_MNIST, NEW_DATA_MNIST
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import csv
import pandas as pd
from collections import deque


input_node_num = 16
input_num_mul_index = 16 #  把输入维度放大10倍



def ysc_create_data_before_pretrain_new_new():
    rows = 10000
    data = []
    groups = {
        'ANGRY': [15, 11],
        'NEGATIVE': [14, 7, 6, 5, 3, 1], # 这里的一个很大的假设就是,如果一起训练可以消极, 那么单个的输入给进来的时候,希望也是消极的!!!!!!
        'POSITIVE': [2, 9, 10, 13],
        'NULL': [0, 4, 8, 12]
        }
    for _ in range(rows):
        selected_group = random.choice(list(groups.values()))
        result_list = [0.0] * input_node_num
        for i in range(input_node_num):
            if i in selected_group:
                result_list[i] = 1.0
            else:
                result_list[i] = random.uniform(0, 0.2)
        data.append(result_list * input_num_mul_index)
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV文件已保存。")


if __name__ == "__main__":
    ysc_create_data_before_pretrain_new_new()
