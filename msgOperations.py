import socket
import utils

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '0.0.0.0' # Here broadcast to all address
PORT = 9999

class scale_info:
    def __init__(self) -> None:
        self.rank_speed_dict = {}
        self.add_overhead = 0
        self.reduce_overhead = 0

class MSGOperations:
    def __init__(self) -> None:
        self.buffer = {}
        self.scale_info = {}

    def get_ranksize_from_msg_str(self, msg):
        if len(msg) <= 0 or "rank_size:" not in msg:
            return
        idx = msg.find("rank_size:")
        tail_str = msg[idx + len("rank_size:"):] # cut string from ranks size
        target_str = tail_str[0:tail_str.find(" ")] # cut string after num of rank size
        return int(target_str)

    # msg_package_dict design
    # key:value 
    # training_speed: dict
    # rank_size: speed
    # add_overhead
    # reduce_overhead
    # Do we need to update the training speed for every step
    def update_rescale_info(self, address, msg):
        if address not in self.buffer:
            return

        msg_list =  self.buffer[address]
        if len(msg_list) < 20: # we assume that first 20 mini steps are not very stable(so ignore the first 20 epoch)
            return

        current_udp_msg = utils.parser_udp_message(msg)
        
        # get training speed base on the msg list
        last_two_msgs = msg_list[len(msg_list) - 2:]
        udp_msgs = [utils.parser_udp_message(ms) for ms in last_two_msgs]
        last_udp_msg = udp_msgs[1]
        second_last_udp_msg = udp_msgs[0]

        time_gap = last_udp_msg.time - second_last_udp_msg.time
        speed = second_last_udp_msg.credit / time_gap
        msg_dict = {}
        msg_dict[second_last_udp_msg.rank_size] = speed

        # get add or reduce overhead
        if current_udp_msg.rank_size > last_udp_msg:
            # Add overhead
            msg_dict["add_overhead"] = current_udp_msg.time - last_udp_msg.time
        else:
            msg_dict["reduce_overhead"] = current_udp_msg.time - last_udp_msg.time

        self.scale_info[address] = msg_dict

    def update_training_speed_info(self, address, msg):
        if address not in self.buffer:
            return
        msg_list =  self.buffer[address]
        if len(msg_list) < 2:
            return

        udp_msg = utils.parser_udp_message(msg)
        # Get training speed

        last_two_msgs = msg_list[len(msg_list) - 2:]
        udp_msgs = [utils.parser_udp_message(ms) for ms in last_two_msgs]
        last_udp_msg = udp_msgs[1]
        second_last_udp_msg = udp_msgs[0]

        time_gap = last_udp_msg.time - second_last_udp_msg.time
        speed = second_last_udp_msg.credit / time_gap
        msg_dict = {}
        msg_dict[second_last_udp_msg.rank_size] = speed

        msg_dict = self.scale_info[address]
        self.scale_info[address] = msg_dict

    # Sever - manager side
    def create_msg_server(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            print("create udp server success")
            w = open("msg.log", "w")
            
            rank = 0
            
            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                msg = address_id + str(data, encoding = "utf-8")
                #print(msg)
                current_rank = self.get_ranksize_from_msg_str(msg)

                if rank != current_rank:
                    # rank change happen
                    # Do something
                    self.update_rescale_info(addr[0], msg)

                    rank = current_rank # update rank

                # get the message buffer info
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
