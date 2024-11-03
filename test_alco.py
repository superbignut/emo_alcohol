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

if __name__=="__main__":
    # 设置串口参数
    port = plist[0][0]  # 串口设备路径
    baudrate = 9600  # 波特率
    ser = serial.Serial(port, baudrate, timeout=1)
    try:
        while True:
            # 从串口读取数据
            # message = ser.readline().decode('utf-8').strip()  # 读取并解码
            message = ser.readline().strip()

            if message:
                
                parsed_values = [byte for byte in message] 

                if len(parsed_values) == 9:
                    a = ord(parsed_values[4]) * 256 + ord(parsed_values[5])
                    b = ord(parsed_values[6]) * 256 + ord(parsed_values[7])
                    print "alcohol is: {}/{}".format(a, b)
    except:
        ser.close()



