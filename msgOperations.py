import socket
import utils

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '10.3.0.108'
if utils.is_theta_cluster():
    ADDRESS = '172.23.2.202' # thetagpu14

PORT = 9999
class MSGOperations:
    def __init__(self) -> None:
        self.buffer = []
    # Sever - manager side
    def create_udp_server(self):
        try:
            print("create udp server")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                msg = address_id + str(data, encoding = "utf-8")
                self.buffer.append(msg)
                print(msg)
        except Exception as ex:
            print("Create Server failed")
            print(ex)
            s.close()
    # Client
    def create_udp_client(self, address, port):
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
