import socket
import utils

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '0.0.0.0' # Here broadcast to all address
PORT = 9999

class scale_info:
    def __init__(self, id) -> None:
        self.jobid = id
        self.rank_speed_dict = {}
        self.add_overhead = 0
        self.reduce_overhead = 0

class MSGOperations:
    def __init__(self) -> None:
        self.buffer = {}
        self.scale_info_dict = {}

    def get_ranksize_from_msg_str(self, msg):
        if len(msg) <= 0 or "rank_size:" not in msg:
            return
        idx = msg.find("rank_size:")
        tail_str = msg[idx + len("rank_size:"):] # cut string from ranks size
        target_str = tail_str[0:tail_str.find(" ")] # cut string after num of rank size
        return int(target_str)

    def _get_training_speed(self, last_udp, penul_udp):
        time_gap = last_udp.time - penul_udp.time
        speed = (last_udp.credit + penul_udp.credit)/ 2 / time_gap
        return speed

    def _get_overhead(self, msg_udp, last_udp, penul_udp):
        time_gap_before_rank_change = last_udp.time - penul_udp.time
        time_gap_on_rank_change = msg_udp.time - last_udp.time
        overhead = time_gap_on_rank_change - time_gap_before_rank_change
        return overhead

    def update_scale_info_dict(self, address, msg):
        """Update rescale info dict only when rank change"""
        
        # Here means the job is a new started job and no training information in the buffer that we could use to 
        # cal training speed and add reduce overhead, so there is no meaning to keep on
        if address not in self.buffer:
            return
        
        msg_list =  self.buffer[address]
        if len(msg_list) < 10: # we assume that first 10 mini steps are not very stable(so ignore the first 10 mini step)
            return

        msg_udp = utils.parser_udp_message(msg)
        jobid = msg_udp.id
        
        last_two_msgs = msg_list[-2:]
        last_udp = utils.parser_udp_message(last_two_msgs[-1])
        penultimate_udp = utils.parser_udp_message(last_two_msgs[0])

        training_speed = self._get_training_speed(last_udp, penultimate_udp)
        overhead = self._get_overhead(msg_udp, last_udp, penultimate_udp)

        # if scale info existed then update
        if jobid in self.scale_info_dict:
            job_scale_info  = self.scale_info_dict[jobid]
            job_scale_info.rank_speed_dict[last_udp.rank_size] = training_speed

            if overhead > 0:
                if msg_udp.rank_size > last_udp.rank_size:
                    job_scale_info.add_overhead = overhead
                else:
                    job_scale_info.reduce_overhead = overhead
        else: # if not then create a new info instance
            job_scale_info = scale_info(jobid)
            job_scale_info.rank_speed_dict[last_udp.rank_size] = training_speed
            
            if overhead > 0:
                if msg_udp.rank_size > last_udp.rank_size:
                    job_scale_info.add_overhead = overhead
                else:
                    job_scale_info.reduce_overhead = overhead

            self.scale_info_dict[jobid] = job_scale_info

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
                    # rank change happen update the scale info dict
                    self.update_scale_info_dict(addr[0], msg)
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
