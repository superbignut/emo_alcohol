from pydub import AudioSegment

# 加载 MP3 文件
mp3_file = "./music.mp3"
audio = AudioSegment.from_mp3(mp3_file)

# 转换为 WAV 格式
wav_file = "music.wav"
audio.export(wav_file, format="wav")

print("转换完成！")