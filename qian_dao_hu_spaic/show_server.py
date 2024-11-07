
"""
    这边的话，容易报错是其实还是由于之前运行的时候没把进程杀干净， 
    kill掉就不报错了，我猜报错核心dump 就是 两个进程 同时写文件导致的
"""


import pyaudio
import wave
import time
from zhipuai import ZhipuAI
import pyttsx3
import socket
import base64
import urllib
import requests
import json
import os
# from filelock import FileLock
import traceback
import sys
from Controller import Controller
import threading

API_KEY = "uXF2wBd5nWGfay9qfJzhkPO3"
SECRET_KEY = "3bghdtbtwYc1M0FINptHjz5fEZNVjvpe"


def _bo_fang(index):
    # with audio_lock: # 在多线程之前上锁
    try:
        if index == 1:
            file_name = "wang_wang.wav"
        elif index == 2:
            file_name = "woof_sad.wav"
        elif index == 3:
            file_name = "he_le_hen_duo_le.wav"
        elif index == 4:
            file_name = "3wang.wav"
        elif index == 5:
            file_name = "music30.wav"
        elif index == 6:
            file_name = "buxihuan_hongse.wav"
        # 打开.wav文件
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        wf = wave.open(file_path, 'rb')

        # 创建PyAudio对象
        p = pyaudio.PyAudio()

        # 打开音频流
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # 播放数据
        data = wf.readframes(1024)

        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # 停止音频流
        stream.stop_stream()
        stream.close()

        # 关闭 PyAudio
        p.terminate()
        wf.close()
    except:
        print("error")
    finally:
        print("over")



file_path = "output.wav"

def baidu_wav_to_words():

    url = "https://vop.baidu.com/server_api"

    speech = get_file_content_as_base64(file_path, False)
    payload = json.dumps({
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "vks6nBUXlchi2SekxmPHOuFoqW0UpeMe",
        "dev_pid": 1537,
        "speech": speech,
        "len": os.path.getsize(file_path),
        "token": get_access_token()
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)
    return(response.json().get('result')[0])


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))



def demo_api(input_txt):
    conversation_id = None
    output=input_txt
    api_key = "299adac92d9b98c139f22fa1e22a8b2c.t7LzNyfNX49gsShG"
    url = "https://open.bigmodel.cn/api/paas/v4"
    client = ZhipuAI(api_key=api_key, base_url=url)
    prompt = output
    generate = client.assistant.conversation(
        assistant_id="659e54b1b8006379b4b2abd6",
        conversation_id=conversation_id,
        model="glm-4-assistant",
        messages=[
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }
        ],
        stream=True,
        attachments=None,
        metadata=None
    )
    output = ""
    for resp in generate:
        if resp.choices[0].delta.type == 'content':
            output += resp.choices[0].delta.content
            conversation_id = resp.conversation_id
    return output


def lu_yin_and_save():
    print("yes1")
    # 配置录音参数
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2 # 录音的时间
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    device_name = 'default'  # 你想使用的设备名称
    device_index = None
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if device_name in info['name']:
            device_index = i
            break
    print("yes1")

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=device_index)

    print("录音中...")

    frames = []

    # 录制音频
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # 如果 锁来了， 就break
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束")

    # 停止和关闭音频流

    
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 将录制的音频保存为wav文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("yes3")
    time.sleep(1)

def bo_fang():
    wf = wave.open("output.wav", 'rb')  

    # 创建PyAudio对象
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 播放数据
    data = wf.readframes(1024)

    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # 停止音频流
    stream.stop_stream()
    stream.close()

    # 关闭 PyAudio
    p.terminate()
    wf.close()

    time.sleep(1)

class Gouzi:

    def __init__(self) -> None:

        #### 全局变量， 从socket线程中修改， 用于 self.net 的前向推理
        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
        self.color = 0 #  0 1 2  无， 蓝色， 红色               3
        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
        self.dmx = 0 # 0 1 2 3 # 无，积极，消极                  3
        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2
        self.emo_len = 1

        self.controller = None
        
        self.action_socket_init() # 初始化controller

        
        self.state_number = 0 # 
        self.is_moving = False
        """
            0 趴下
            1 起立
            2 检测其他的内容
        
        """

    def action_socket_init(self):
        # client_address = ("192.168.1.103", 43897)
        server_address = ("192.168.1.120", 43893)  # 运动主机端口
        self.controller = Controller(server_address) # 创建 控制器

        self.controller.heart_exchange_init() # 初始化心跳
        time.sleep(2)
        # self.controller.stand_up()
        # print('stand_up')
        # pack = struct.pack('<3i', 0x21010202, 0, 0)
        # controller.send(pack) # 
        self.controller.not_move() # 进入 静止状态
        print("动作socket初始化")
    
    def dmx_qili(self):
        self.controller.do_move()
        self.say_something(index=1) # 汪一下
        time.sleep(0.5)
        self.controller.stand_up()
        self.controller.not_move()
        # self.say_something(index=1) # 汪一下

    def zou_guo_lai(self):
        self.controller.do_move()
        self.say_something(index=1) # 汪一下
        time.sleep(0.5)
        self.controller.fuyang_or_qianhou()
        time.sleep(3)
        self.controller.thread_active = False # 结束 前进
        self.controller.not_move()

    def _start_server_thread(self, host='192.168.1.103', port=12345):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5) # 等待数， 连接数
        print(f"Server listening on {host}:{port}...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            client_handler = threading.Thread(target=self.handle_client_thread, args=(client_socket,))
            client_handler.start()
    def start_server(self, host='192.168.1.103', port=12345):
        temp_thread = threading.Thread(target=self._start_server_thread, name="_start_server_thread")
        temp_thread.start()


    def say_something(self, index):    
        #  temp_p = multiprocessing.Process(target=_bo_fang, args=(index, ))
        temp_t = threading.Thread(target=_bo_fang, name="bo_fang_thread", args=(index, ))
        temp_t.start() # 这里 因为麦克风是 io 且是独占的，所有 多线程可以加速， 并且 需要join
        temp_t.join()
        # _bo_fang(index=index) # 干掉多线程
        print("播放结束")

    def do_action_from_input(self):
        # 这里只要状态使用过一次之后 ， 就会被 置零
        while True:

            if self.imu == 1:
                
                self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################

                self.controller.zuo_you_huang()
                self.say_something(index=4) # 3wang
                
                time.sleep(1)
                self.controller.thread_active = False
                
                #########################################
                # 执行完所有动作后
                self.is_moving = False
                ########################################
                
            elif self.imu == 2:
                self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################

                self.controller.pian_hang()
                self.say_something(index=2)
                time.sleep(1)
                self.controller.thread_active = False

                #########################################
                # 执行完所有动作后
                self.is_moving = False
                ########################################
            
            elif self.color == 2:
                self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################
                
                self.controller.low_height_of_dog() # 红衣服 半蹲
                time.sleep(1)
                self.controller.thread_active = False

                #########################################
                # 执行完所有动作后
                self.is_moving = False
                ########################################
            if self.alcohol == 1:
                
                self.controller.light_eyes() # 等待的时候闪闪眼睛 或者
                time.sleep(1)
                #########################################
                self.controller.thread_active = False # 先关掉所有的动作
                self.is_moving = True 
                #########################################
                self.say_something(index=3)

                self.say_something(index=5)
                time.sleep(1)

                self.controller.thread_active = False

                #########################################
                # 执行完所有动作后
                self.is_moving = False
                #######################################

            self.clear()
            time.sleep(1) #  统统给我 等待一秒

            
    def clear(self):
        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
        self.color = 0 #  0 1 2  无， 蓝色， 红色               3
        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
        self.dmx = 0 # 0 1 2 3 # 无，积极，消极                  3
        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2



    def handle_client_thread(self, client_socket):
        # 处理线程
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                # print(data)
                command, args1, args2 = data.decode('utf-8').split()  # 假设数据格式为 "COMMAND arg1 arg2"
                # print(command, args1, args2)
                args1 = int(args1)
                # args2 = int(args2)
                print(f"Received command: {command}, args: {args1}, {args2}")

                """
                        self.imu = 0 #  0 1 2     无、抚摸， 抚摸+、敲打、敲打+   5
                        self.color = 0 #  0 1 2   无， 蓝色， 红色               3
                        self.alcohol = 0 #  0 1 2   无， 酒精 酒精               3
                        self.dmx = 0 #    1 2 3 #  积极，消极, 正常              3
                        self.gesture = 0 # 0 1 2 3 无  上，下，挥手              4
                        self.power = 0 # 0 1 # 20% 以下， 和以上的电量           2
                """
                if self.is_moving:
                    time.sleep(0.2)
                    return

                if command == "gesture":
                    if args1 == 4: #  nice 表扬 手势
                        self.gesture = 1
                        print("up_gesture")
                    elif args1 == 5: #  批评手势
                        print("down_gesture") 
                        self.gesture = 2
                    elif args1 == 1: # 手掌手势
                        print("hello_gesture") 
                        self.gesture = 3
                    else:
                        self.gesture = 0


                elif command == "imu":
                    if args1 == 1: # 抚摸
                        print("touching_imu")
                        self.imu = 1
                    elif args1 == 2: # 敲打
                        print("was kicked_imu")
                        self.imu = 2
                    else:
                        self.imu = 0


                elif command == "color":
                    if args1 == 1:
                        print("red_color")
                        self.color = 2 
                    else:
                        self.color = 0


                elif command == "alco":
                    if args1 == 1:
                        print("drink_alco")
                        self.alcohol = 1
                    else:
                        self.alcohol = 0


                    """                     elif command == "dmx":
                    if args1 == 1:
                        print("biao_yang_dmx")
                        self.dmx = 1
                    elif args1 == 2:
                        print("pi_ping_dmx")
                        self.dmx = 2 
                    elif args1 == 3:
                        print("nothing_dmx")
                        self.dmx = 3 """


                elif command == "power":
                    if args1 == 1:
                        self.power = 1
                    else:
                        self.power = 0 # 正常
                    

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))

        finally:
            client_socket.close()




def check_state_0(dog):
    while dog.state_number == 0: # 最开始的状态是0
        try:
            lu_yin_and_save() # wav

            text = baidu_wav_to_words() # wav 
            print(text) 

            text  = "从一个小狗的角度，判断下面这段话属于，1让我站起来，2让我走过去，3其他内容， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
            print(text)
            
            # bo_fang()

            web_text = demo_api(input_txt=text) # 

            print(web_text)

            if web_text == "1":
                dog.state_number = 1
                dog.dmx_qili()
                break

            time.sleep(0.5) # 大模型不是一直在录音 也会有间断的
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))
            print("error")
            continue
def check_state_1(dog):
    while dog.state_number == 1: # 最开始的状态是0
        try:
            lu_yin_and_save() # wav

            text = baidu_wav_to_words() # wav 
            print(text) 

            text  = "从一个小狗的角度，判断下面这段话属于，1让我站起来，2让我走过去，3其他内容， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
            print(text)
            
            # bo_fang()

            web_text = demo_api(input_txt=text) # 

            print(web_text)

            if web_text == "2":
                dog.state_number = 2
                dog.zou_guo_lai()
                time.sleep(3)
                dog.clear()#  这里有一个等待和清空
                break

            time.sleep(0.5) # 大模型不是一直在录音 也会有间断的
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
            # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
            # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
            print(traceback.format_exception(
                exc_type,
                exc_value,
                exc_traceback
            ))
            print("error")
            continue
def run():

    ysc_dog=Gouzi()
    ysc_dog.start_server() #  启动任务的监视进程
    # dog.state_number = -1
    check_state_0(dog=ysc_dog)
    check_state_1(dog=ysc_dog)
    ysc_dog.do_action_from_input() # 开始循环了

if __name__ == '__main__':
    run()