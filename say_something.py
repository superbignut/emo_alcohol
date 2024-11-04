import multiprocessing
import pyaudio
import os
import wave

def say_something(index):
    temp_p = multiprocessing.Process(target=bo_fang, args=(index, ))
    temp_p.start()
    print("播放结束")

    # 这里可以单独启动一个线程 ，如果某个状态 一直不变，那么就置零

def bo_fang(index):
    print(type(index), index)
    if index == 1:
        file_name = "wang_wang.wav"
    elif index == 2:
        file_name = "woof_sad.wav"
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
        #print(data)
        stream.write(data)
        data = wf.readframes(1024)

    # 停止音频流
    stream.stop_stream()
    stream.close()

    # 关闭 PyAudio
    p.terminate()


if __name__ == '__main__':
    say_something(1)