from manager import Manager
import time

class MessageOperator:
    def __init__(self, address, port) -> None:
        self.address = address
        self.port = port
        self.socket = Manager().create_msg_client(address, port)
        self.sequence_id = 0

    def report(self, credit, rank_size):
        t = time.time()
        report_msg = 'id:%d time:%f rank_size:%d credit:%s' % (self.sequence_id, t, rank_size, credit)
        self.sequence_id += 1
        try:
            self.socket.sendto(str.encode(report_msg), (self.address, self.port))
        except Exception as ex:
            self.socket.close()
