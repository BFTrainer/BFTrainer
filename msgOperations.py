import socket

ADDRESS = '10.3.0.108'
PORT = 9999

class MSGOperations:

    def __init__(self) -> None:
        self.buffer = []

    # Sever - manager side
    def create_udp_server(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                msg = address_id + str(data, encoding = "utf-8")
                self.buffer.append(msg)
                print(msg) # report to terminal
        except Exception as ex:
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
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return s
        except Exception as ex:
            s.close()
