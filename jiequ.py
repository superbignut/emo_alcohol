import wave
import pyaudio

# 输入和输出 WAV 文件路径
input_filename = 'input.wav'
output_filename = 'output_30s.wav'

# 打开原始 WAV 文件
with wave.open(input_filename, 'rb') as in_file:
    # 获取音频的帧率（采样率）和每秒的帧数
    framerate = in_file.getframerate()
    total_frames = in_file.getnframes()

    # 计算截取的帧数（前 30 秒的帧数）
    max_frames_to_capture = framerate * 30  # 30 秒的帧数
    frames_to_read = min(max_frames_to_capture, total_frames)

    # 读取前 30 秒的音频数据
    audio_data = in_file.readframes(frames_to_read)

# 创建新的 WAV 文件以保存前 30 秒的音频
with wave.open(output_filename, 'wb') as out_file:
    # 设置与原文件相同的参数
    out_file.setnchannels(in_file.getnchannels())  # 通道数
    out_file.setsampwidth(in_file.getsampwidth())  # 采样宽度
    out_file.setframerate(framerate)               # 采样率
    out_file.writeframes(audio_data)               # 写入音频数据

print(f"截取的前 30 秒已保存为 {output_filename}")
