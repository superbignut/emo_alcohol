# -*- coding: utf-8 -*-
from Controller import Controller
import time
import socket


client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)  # 运动主机端口
controller = Controller(server_address) # 创建 控制器

controller.heart_exchange_init() # 初始化心跳
controller.stand_up()
#self.controller.stand_up()
# pack = struct.pack('<3i', 0x21010202, 0, 0)
# controller.send(pack) # 
time.sleep(2)

controller.not_move() # 进入 静止状态
controller.light_eyes() # 眼睛发光
time.sleep(5)

controller.thread_active = False
controller.heart_flag = False

print("结束")