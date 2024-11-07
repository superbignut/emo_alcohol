# -*- coding: utf-8 -*-
"""
    可以成功说出话了
"""


import wave
import pyaudio

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
    stream.write(data)
    data = wf.readframes(1024)

# 停止音频流
stream.stop_stream()
stream.close()

# 关闭 PyAudio
p.terminate()