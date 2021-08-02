import socket

ADDRESS = '10.3.0.108'
PORT = 9999

class MSGOperations:

    def __init__(self) -> None:
        self.buffer = []

    # Sever - manager side
    def create_udp_server(self):
        try:
            print("message server created")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                self.buffer.append(address_id + str(data, encoding = "utf-8"))
        except Exception as ex:
            print("udp server creation error message: %s" % ex)
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
            s.sendto(b'Client Successfully Created', (address, port))
            return s
        except Exception as ex:
            print("udp client creation error message: %s" % ex)
            s.close()
