# from manager import Manager
import time, socket

def create_msg_client(address, port):
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

class MessageOperator:
    def __init__(self, address, port) -> None:
        self.address = address
        self.port = port
        self.socket = create_msg_client(address, port)
        self.sequence_id = 0

    def report(self, credit, rank_size, jobname):
        t = time.time()
        report_msg = 'id:%d time:%f rank_size:%d credit:%s jobname:%s' % (self.sequence_id, t, rank_size, credit, jobname)
        self.sequence_id += 1
        try:
            self.socket.sendto(str.encode(report_msg), (self.address, self.port))
        except Exception as ex:
            self.socket.close()
