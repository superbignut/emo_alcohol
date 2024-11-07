import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0:
        print(f"Device Index: {i}, Device Name: {dev_info['name']}")

p.terminate()