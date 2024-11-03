import pyaudio
import wave
import time
from baidu import baidu_wav_to_words
from demo import demo_api
from shuo_hua import shuo_zhong_wen
from Controller import Controller


import socket
host = '192.168.1.103'
socket_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((host, socket_port))


def lu_yin_and_save():

    # 配置录音参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    # 打开音频流
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=28)

    print("录音中...")

    frames = []

    # 录制音频
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        #print(type(data))
        # break
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

""" def load_and_bo_fang():
    # -*- coding: utf-8 -*-


    # 打开.wav文件
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
        # print(type(data))
        stream.write(data)
        data = wf.readframes(1024)

    # 停止音频流
    stream.stop_stream()
    stream.close()

    # 关闭 PyAudio
    p.terminate() """

def run():
    client_address = ("192.168.1.103", 43897)
    server_address = ("192.168.1.120", 43893)  # 运动主机端口
    controller = Controller(server_address) # 创建 控制器

    controller.heart_exchange_init() # 初始化心跳
    print("起立")
    # controller.stand_up()
    # pack = struct.pack('<3i', 0x21010202, 0, 0)
    # controller.send(pack) # 起立
    time.sleep(4)

    controller.not_move()

    while True:
        try:
            # controller.change_damoxing_flag()

            # controller.tai_tou()  # 抬头

            lu_yin_and_save()
            # load_and_bo_fang()

            text = baidu_wav_to_words()
            print(text)

            # controller.change_damoxing_flag()
            # controller.di_tou()  # 低头

            # shuo_zhong_wen("你别急.")
            text  = "从一个小狗的角度，判断下面这段话属于，1表扬我，2批评我，3正常谈话， 当中的哪个种类，只需要用编号1或2或3来回答我。" + text
            print(text)
            web_text = demo_api(input_txt=text)
            # if len(web_text) < 2:
            #     web_text = "大模型没有输出啊。"
            print(web_text)

            data = "dmx " + web_text + " " + str(0)
            client_socket.sendall(data.encode('utf-8'))
            # shuo_zhong_wen(web_text)
            time.sleep(0.1)
        except:
            time.sleep(0.1)
            continue




if __name__ == '__main__':
    """     lu_yin_and_save()
    time.sleep(1)
    load_and_bo_fang()
    time.sleep(1)
    text = baidu_wav_to_words()
    print(text)
    web_text = demo_api(input_txt=text)
    print(web_text) """
    run()