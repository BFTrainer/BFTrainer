import socket
from queue import Queue
from typing import List

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '0.0.0.0' # Here broadcast to all address
PORT = 9999

class MSGOperations:
    def __init__(self) -> None:
        self.buffer = {}

    # Sever - manager side
    def create_msg_server(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            print("create udp server success")
            w = open("msg.log", "w")
            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                msg = address_id + str(data, encoding = "utf-8")
                #print(msg)

                if addr[0] in self.buffer:
                    tmp_que = self.buffer[addr[0]]
                    if len(tmp_que) >= 100:
                        tmp_que.pop(0)
                    tmp_que.append(msg)
                else:
                    q = []
                    q.append(msg)
                    self.buffer[addr[0]] = q

                w.write(msg + '\n')
                w.flush()
                # print(msg)
                # print("queue size: ", self.buffer.qsize())
        except Exception as ex:
            print("Create UDP Server failed")
            print(ex)
            s.close()

    # Client
    def create_msg_client(self, address, port):
        """Create udp client
        Args:
            address (str): msg address
            port (int): msg port
        Returns:
            [socket]: [udp client socket]
        """
        try:
            print("create udp")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return s
        except Exception as ex:
            print("Create client failed")
            s.close()
