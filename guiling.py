# 机器狗起立前校准
import time
import struct
import threading
import os
import psutil
from Controller import Controller

# global config
client_address = ("192.168.1.103", 43897)
server_address = ("192.168.1.120", 43893)
# creat a controller
controller = Controller(server_address)



# start to exchange heartbeat pack
def heart_exchange(con):
    pack = struct.pack('<3i', 0x21040001, 0, 0)
    while True:
        con.send(pack)
        time.sleep(0.25)  # 4Hz
heart_exchange_thread = threading.Thread(target=heart_exchange, args=(controller,))
heart_exchange_thread.start()

# stand up
pack = struct.pack('<3i', 0x21010C05, 0, 0)
controller.send(pack)