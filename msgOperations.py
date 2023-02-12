import socket, time
import utils

# This address is for 
# ADDRESS = '172.23.2.202' # thetagpu14

ADDRESS = '0.0.0.0' # Here broadcast to all address
PORT = 5555

class scale_info:
    def __init__(self, id) -> None:
        self.jobid = id
        self.rank_speed_dict = {}
        self.add_overhead = 0
        self.reduce_overhead = 0

class MSGServer:
    def __init__(self) -> None:
        self.buffer = {}
        self.scale_info_dict = {} # keys:jobnames value:scale_info instance

    def get_jobid_and_ranksize_from_msg(self, msg):
        if len(msg) <= 0 or "rank_size:" not in msg:
            return

        jobname_idx = msg.find("jobname:")
        jobname_str = msg[jobname_idx + len("jobname:"):]
        
        idx = msg.find("rank_size:")
        tail_str = msg[idx + len("rank_size:"):] # cut string from ranks size
        target_str = tail_str[0:tail_str.find(" ")] # cut string after num of rank size
        return jobname_str, int(target_str)

    def _get_training_speed(self, last_udp, penul_udp):
        time_gap = last_udp.time - penul_udp.time
        speed = (last_udp.credit + penul_udp.credit)/ 2 / time_gap
        return speed

    def _get_overhead(self, msg_udp, last_udp, penul_udp):
        time_gap_before_rank_change = last_udp.time - penul_udp.time
        time_gap_on_rank_change = msg_udp.time - last_udp.time
        overhead = time_gap_on_rank_change - time_gap_before_rank_change
        return overhead

    def update_scale_info_dict(self, jobname, msg, file):
        """Update rescale info dict only when rank change"""
        utils.print_colored_log(f"job {jobname} update scale info dict func called", color="GREEN")

        # Here means the job is a new started job and no training information in the buffer that we could use to 
        # cal training speed and add reduce overhead, so there is no meaning to keep on
        if jobname not in self.buffer:
            return
        
        msg_list =  self.buffer[jobname]
        if len(msg_list) < 10: # we assume that first 10 mini steps are not very stable(so ignore the first 10 mini step)
            return

        msg_udp = utils.parser_udp_message(msg)
        
        last_two_msgs = msg_list[-2:]
        last_udp = utils.parser_udp_message(last_two_msgs[-1])
        penultimate_udp = utils.parser_udp_message(last_two_msgs[0])

        training_speed = self._get_training_speed(last_udp, penultimate_udp)
        overhead = self._get_overhead(msg_udp, last_udp, penultimate_udp)

        # if scale info existed then update
        if jobname in self.scale_info_dict:
            job_scale_info  = self.scale_info_dict[jobname]
            job_scale_info.rank_speed_dict[last_udp.rank_size] = training_speed

            if overhead > 0:
                if msg_udp.rank_size > last_udp.rank_size:
                    job_scale_info.add_overhead = overhead
                else:
                    job_scale_info.reduce_overhead = overhead
        else: # if not then create a new info instance
            job_scale_info = scale_info(jobname)
            job_scale_info.rank_speed_dict[last_udp.rank_size] = training_speed
            
            if overhead > 0:
                if msg_udp.rank_size > last_udp.rank_size:
                    job_scale_info.add_overhead = overhead
                else:
                    job_scale_info.reduce_overhead = overhead
            self.scale_info_dict[jobname] = job_scale_info

        utils.print_colored_log(f"jobname is {jobname} and job_scale_info keys:{job_scale_info.rank_speed_dict.keys()} values {job_scale_info.rank_speed_dict.values()} job add overhead {job_scale_info.add_overhead} job reduce overhead {job_scale_info.reduce_overhead}", color="GREEN")
        file.write(f"jobname is {jobname} and job_scale_info keys:{job_scale_info.rank_speed_dict.keys()} values {job_scale_info.rank_speed_dict.values()} job add overhead {job_scale_info.add_overhead} job reduce overhead {job_scale_info.reduce_overhead}\n")

    # Sever - manager side
    def create_msg_server(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((ADDRESS, PORT))
            print(f"create udp server success, address is {ADDRESS} and port is {PORT}")
            w = open("msg.log", "w")

            # This is a temp way to detect the rank change method
            job_current_rank_dict = {}

            while True:
                data, addr = s.recvfrom(1024)
                address_id = 'Address:%s ' % addr[0]
                msg = address_id + str(data, encoding = "utf-8")
                # print(msg)
                jobname, current_rank = self.get_jobid_and_ranksize_from_msg(msg)

                stored_rank = 0
                # get old rank
                if jobname in job_current_rank_dict:
                    stored_rank = job_current_rank_dict[jobname]

                if stored_rank != current_rank:
                    w.write("Rank change triggered\n")
                    w.write(f"stored rank: {stored_rank} current_rank: {current_rank}\n")
                    utils.print_colored_log(f"[information from msgOperations.py]: Rank change detected, job {jobname} previous rank: {stored_rank} current_rank: {current_rank}", color="GREEN")
                    self.update_scale_info_dict(jobname, msg, w)

                    stored_rank = current_rank # update rank
                    job_current_rank_dict[jobname] = stored_rank

                # get the message buffer info
                if jobname in self.buffer:
                    tmp_que = self.buffer[jobname]
                    if len(tmp_que) >= 100:
                        tmp_que.pop(0)
                    tmp_que.append(msg)
                else:
                    q = []
                    q.append(msg)
                    self.buffer[jobname] = q

                w.write(msg + '\n')
                w.flush()
        except Exception as ex:
            print("Create UDP Server failed")
            print(ex)
            s.close()

class MSGClient:
    def __init__(self, address, port) -> None:
        self.address = address
        self.port = port
        self.socket = self.create_msg_client()
        self.sequence_id = 0

    def report(self, credit, rank_size, jobname):
        t = time.time()
        report_msg = 'id:%d time:%f rank_size:%d credit:%s jobname:%s' % (self.sequence_id, t, rank_size, credit, jobname)
        self.sequence_id += 1
        try:
            self.socket.sendto(str.encode(report_msg), (self.address, self.port))
        except Exception as ex:
            self.socket.close()

    def create_msg_client(self):
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
