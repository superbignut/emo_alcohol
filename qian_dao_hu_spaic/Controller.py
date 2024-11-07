import socket
import struct
import threading
import time
import psutil
import os
class Controller:
    def __init__(self, dst):
        self.lock = False # 用于每次一个动作
        self.last_ges = "stop" # 上一个动作是什么
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = dst # 目标 地址
        self.stop_heartbeat = False
        self.thread_active = False
        self.move_mode = False

        self.damoxing_flag = True

        self.heart_flag = True # 最后的最后 关闭心跳

    def change_damoxing_flag(self):
        self.damoxing_flag = False # 先
        time.sleep(0.1)
        self.damoxing_flag = True    

    


    def heart_exchange_init(self):
        # start to exchange heartbeat pack
        def heart_exchange(con):
            pack = struct.pack('<3i', 0x21040001, 0, 0)
            while self.heart_flag:
                if self.stop_heartbeat:
                    return
                con.send(pack)
                time.sleep(0.2)  # 4Hz
        heart_exchange_thread = threading.Thread(target=heart_exchange, args=(self,))
        heart_exchange_thread.start()
        # used to send a pack to robot dog

    def send(self, pack):
        self.socket.sendto(pack, self.dst)

    def stand_up(self):
        pack = struct.pack('<3i', 0x21010202, 0, 0) # 打招呼是ok的
        self.send(pack)
        if self.last_ges == "stop":
            self.last_ges = "stand"

        elif self.last_ges == "lie_down":
            self.last_ges = "stand"

        elif self.last_ges == "stand":
            self.last_ges = "lie_down"


    def stop_all_actions(self):
        self.send(struct.pack('<3i', 0x21010130, 0, 0))
        self.send(struct.pack('<3i', 0x21010131, 0, 0))
        self.send(struct.pack('<3i', 0x21010102, 0, 0))
        self.send(struct.pack('<3i', 0x21010135, 0, 0))

    def tai_tou(self):
        print("抬头了！！！")
        self.not_move()
        def temp_thread():
            while self.damoxing_flag:
                pack = struct.pack('<3i', 0x21010130, -20005, 0)
                self.send(pack) # 
                time.sleep(0.1)
        tt = threading.Thread(target=temp_thread, name="tai_tou_thread_ltl")
        tt.start()
            

    def di_tou(self):
        print("低头了！！！")
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        self.not_move()
        def temp_thread():
            while self.thread_active:
                pack = struct.pack('<3i', 0x21010130, 13000, 0) # 就是放平得了
                self.send(pack) # 
                time.sleep(0.1)
        tt = threading.Thread(target=temp_thread, name="di_tou_thread_ltl")
        tt.start()
        
    def hou_tui_2s(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        self.do_move()
        time.sleep(0.1)
        def temp_func():
            print("开始后退")
            while self.thread_active:
                pack = struct.pack('<3i', 0x21010130, -6555, 0)
                self.send(pack) # 
                time.sleep(0.3)

        temp_thread = threading.Thread(target=temp_func, name="hou_tui_thread")
        temp_thread.start()
        time.sleep(1)
        
        self.thread_active = False
        self.not_move()  # 关闭并切回




    def not_move(self):
        # 
        self.move_mode = False
        self.send(struct.pack('<3i', 0x21010D05, 0, 0))
        self.send(struct.pack('<3i', 0x21010D05, 0, 0)) # 发送两次
        

    def do_move(self):
        self.move_mode = True
        self.send(struct.pack('<3i', 0x21010D06, 0, 0))
        # time.sleep(0.1)
        self.send(struct.pack('<3i', 0x21010D06, 0, 0))
        

    def fuyang_or_qianhou(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        def temp_func():
            print("开始摇摆")
            while self.thread_active:
                if not self.move_mode: # 不是运动模式
                    pack = struct.pack('<3i', 0x21010130, 13000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)
                    pack = struct.pack('<3i', 0x21010130, -13000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)    
                else:
                    pack = struct.pack('<3i', 0x21010130, 6554, 0)
                    self.send(pack) # 
                    time.sleep(0.3)

        temp_thread = threading.Thread(target=temp_func, name="temp_func_in_fuyang")
        temp_thread.start()
    
    def low_height_of_dog(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        def temp_func():
            print("降低")
            while self.thread_active:
                if not self.move_mode: # 不是运动模式
                    pack = struct.pack('<3i', 0x21010102, -28000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)
        pack = struct.pack('<3i', 0x21010102, 0, 0)
        self.send(pack) # 回复初始高度
        temp_thread = threading.Thread(target=temp_func, name="temp_func_gaodu")
        temp_thread.start()

    def zuo_you_huang(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        def temp_func():
            print("开始摇摆")
            while self.thread_active:
                if not self.move_mode: # 不是运动模式
                    pack = struct.pack('<3i', 0x21010131, 20000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)
                    pack = struct.pack('<3i', 0x21010131, -20000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)    

        temp_thread = threading.Thread(target=temp_func, name="temp_func_in_zuoyou")
        temp_thread.start()

    def pian_hang(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        def temp_func():
            print("开始摇摆")
            while self.thread_active:
                if not self.move_mode: # 不是运动模式
                    pack = struct.pack('<3i', 0x21010135, 15000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)
                    pack = struct.pack('<3i', 0x21010135, -15000, 0)
                    self.send(pack) # 
                    time.sleep(0.3)    

        temp_thread = threading.Thread(target=temp_func, name="temp_func_in_pianhang")
        temp_thread.start()



    def light_eyes(self):
        if self.last_ges != "stand": # 必须得在 stand的前提下才行
            return 
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        def _light():
            while self.thread_active:
                # print("yes")
                self.send(struct.pack('<3i', 0x21010507, 0, 0))
                time.sleep(0.2)  # dance 22 seconds then it can process next gesture
        temp_thread = threading.Thread(target=_light)
        temp_thread.start()

    def da_zhao_hu(self):
        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        print(self.last_ges)

        if self.last_ges != "lie_down":
            time.sleep(1)
            self.stand_up() # 要想打招呼,就要先趴下才行

        def _hello():
            i= 1
            while self.thread_active:
                self.lock = True
                print("yes")
                self.send(struct.pack('<3i', 0x21010507, 0, 0))
                time.sleep(0.2)  # dance 22 seconds then it can process next gesture
                self.lock = False
                i+=1
                if i> 14:  # i = 14 的时候正好可以 做两组
                    break
            # self.stand_up()


                
        temp_thread = threading.Thread(target=_hello)
        temp_thread.start()
        # temp_thread.join()
        # _hello()


    def drive_dog(self, ges, val = 10000):

        self.thread_active = False # 先关闭， 也就是把其他线程关掉
        time.sleep(0.1)
        self.thread_active = True # 再开启
        self.not_move()

        #  这个函数使用之前不需要  先 进入静止状态吗？
        print(ges)
        if self.lock or ges == self.last_ges:
            return # 动作重复

        if self.last_ges == "squat" and ges != "squat":
            # stand up
            self.send(struct.pack('<3i', 0x21010202, 0, 0)) # 先站起来
        else:
            # stop all actions
            self.send(struct.pack('<3i', 0x21010130, 0, 0)) # 俯仰角 0 
            self.send(struct.pack('<3i', 0x21010131, 0, 0)) # 横滚角 0
            self.send(struct.pack('<3i', 0x21010102, 0, 0)) # 身体高度 0  # 身体高度要清零，可能会有点僵硬？？
            self.send(struct.pack('<3i', 0x21010135, 0, 0)) # 偏航角 0

        if ges == "squat":
            self.send(struct.pack('<3i', 0x21010202, 0, 0)) # 趴下 
        elif ges == "turning":
            def turn_360():
                self.lock = True
                self.send(struct.pack('<3i', 0x21010135, 13000, 0))  # 转弯？？
                time.sleep(2.84)  # need time to turn 360 degrees
                self.send(struct.pack('<3i', 0x21010135, 0, 0))
                self.lock = False

            turn_thread = threading.Thread(target=turn_360, name="turn_360_dog")
            turn_thread.start()
        elif ges == "twisting": #  扭一扭
            def dance():
                while self.thread_active:
                    self.lock = True
                    self.send(struct.pack('<3i', 0x21010204, 0, 0))
                    time.sleep(22)  # dance 22 seconds then it can process next gesture
                    self.lock = False

            twist_thread = threading.Thread(target=dance)
            twist_thread.start()
        elif ges == "hello":
            def hello():
                while self.thread_active:
                    self.lock = True
                    self.send(struct.pack('<3i', 0x21010507, 0, 0))
                    time.sleep(6)  # dance 22 seconds then it can process next gesture
                    self.lock = False

            twist_thread = threading.Thread(target=hello)
            twist_thread.start()


        elif ges == "forward":
            self.send(struct.pack('<3i', 0x21010130, val, 0))
        elif ges == "back":
            self.send(struct.pack('<3i', 0x21010130, -val, 0))
        elif ges == "right":
            self.send(struct.pack('<3i', 0x21010131, val, 0))
        elif ges == "left":
            self.send(struct.pack('<3i', 0x21010131, -val, 0))

        self.last_ges = ges

    def thread_all_stop(self):
        self.thread_active = False
    def stop_and_kill(self):
        self.stop_heartbeat = True # 关闭心跳
        # 杀掉占用最高的进程
        print("kill the biggest python process")
        # 遍历所有进程
        py_procs = {}
        """ for proc in psutil.process_iter():
            # 获取进程信息
            info = proc.as_dict(attrs=['pid', 'name', 'memory_percent'])
            # 如果是python进程，添加到字典中
            if info['name'] in ['python', 'python3']:
                py_procs[info['pid']] = info['memory_percent']
        """
        # 定义一个计数器变量
        count = 1
        # 如果字典不为空且计数器大于0
        while py_procs and count > 0:
            # 找到memory_percent最大的键值对
            max_pid = max(py_procs, key=py_procs.get)
            max_mem = py_procs[max_pid]
            # 杀掉该进程
            os.system(f'sudo kill -9 {max_pid}')
            # 打印结果
            print(f'Killed process {max_pid} with memory_percent {max_mem}')
            # 从字典中删除该键值对
            del py_procs[max_pid]
            # 计数器加一
            count -= 1
