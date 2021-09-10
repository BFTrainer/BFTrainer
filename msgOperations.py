import socket
from queue import Queue

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '0.0.0.0' # thetagpu14
PORT = 9999

class MSGOperations:
    def __init__(self) -> None:
        self.buffer = Queue()

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
                # print(msg)

                # keep only around 500 items in the buffer
                if self.buffer.qsize() <= 500:
                    self.buffer.put(msg)
                else:
                    self.buffer.get() # remove as FIFO
                    self.buffer.put(msg) # put a new one

                w.write(msg + '\n')
                w.flush()
                # print(msg)
                # print("queue size: ", self.buffer.qsize())
        except Exception as ex:
            print("Create Server failed")
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
