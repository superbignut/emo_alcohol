# -*- coding: utf-8 -*-
from time import sleep
import matplotlib.pyplot as plt
import serial
import time
import numpy as np
import random  # 模拟酒精浓度数据
import serial.tools.list_ports
plist = list(serial.tools.list_ports.comports())
print plist[0][0]
ser = serial.Serial(plist[0][0], 9600, timeout=1)
