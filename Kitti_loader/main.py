from Model import model
import argparse
import numpy as np
import os
import sys
import time
import shutil
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import random


def main():
    if not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    else:
        print("current GPU num is", torch.cuda.device_count()) # 返回GPU数目
        print("name of GPU device is", torch.cuda.get_device_name(0)) # 返回GPU名称，设备索引默认从0开始
        print("current using GPU device is", torch.cuda.current_device()) # 返回当前设备索引



























