from itertools import groupby
import socket
from time import sleep
from time import time
import pandas as pd
from enum import Enum
from collections import OrderedDict

import utils
from prog import re_allocate
import psutil
import numpy as np
import managerOperations
from msgOperations import MSGOperations
import sys_admin
from threading import Thread

MAXIMUM_PARALLEL = 2
MONITOR_GAP = 5
NUM_OF_GPUs_PER_NODE = 1

class JobNodeStatus(Enum):
    JOBFAILED=0
    JOBOUT=1
    NODEIN=2
    NODEOUT=3

class MessageOperator:
    def __init__(self, address, port) -> None:
        self.address = address
        self.port = port
        self.socket = Manager().create_msg_client(address, port)
        self.sequence_id = 0

    def report(self, credit, rank_size):
        t = time()
        report_msg = 'id:%d time:%f rank_size:%d credit:%s' % (self.sequence_id, t, rank_size, credit)
        self.sequence_id += 1
        try:
            self.socket.sendto(str.encode(report_msg), (self.address, self.port))
        except Exception as ex:
            self.socket.close()

class Manager:
    def __init__(self, max_parallel = MAXIMUM_PARALLEL, monitor_gap = MONITOR_GAP):
        self.max_parallel = max_parallel
        self.monitor_gap = monitor_gap
        self.create_working_directory()

    current_map = pd.DataFrame()
    # A dictionary for global job information
    job_info_dict = {}

    # create working directory
    def create_working_directory(self):
        managerOperations.create_working_directory()

    # DBOperatioin
    def submit_job(self, min, max, N, O, res_up, res_dw, path):
        """
        Function for user to submit jobs
        """
        return managerOperations.submit_job(min, max, N, O, res_up, res_dw, path)

    def get_job_queue_len(self):
        """
        Function for user to get job queueu length
        """
        return managerOperations.get_job_queue_len()

    def _get_a_job_from_DB(self):
        return managerOperations.get_a_job_from_DB()
    
    # Network Operations
    def create_msg_client(self, address, port):
        """Create a message client by client for reporting training throughput/speed.

        Args:
            func (func pointer): Client need to offer a function to get the real time training speed
        """
        return MSGOperations().create_udp_client(address, port)

    def create_msg_server(self): # pass dynamic update data function into 
        """Create a message server for receive throughput report by client

        Args:
            func (function pointer): a function for deciding when to trigger update job data
        """
        managerOperations.create_msg_server()

    # life cycle functions
    def _managerStart(self):
        sys_nodes=sys_admin.get_avaliable_nodes_from_system()
        print("start mark1")
        if len(sys_nodes) == 0 or (sys_nodes is None):
            print("No nodes avaliable")
            return
        
        if self.get_job_queue_len() == 0:
            print("No valid jobs")
            return
        
        print("start mark2")
        # fetch job from DB
        starting_jobs_num = min(MAXIMUM_PARALLEL, self.get_job_queue_len())
        for i in range(starting_jobs_num):
            job_string = self._get_a_job_from_DB()
            jobdetail = utils.parser_job_string_2_job_item(job_string)
            self.job_info_dict[jobdetail.GUID] = jobdetail
        
        print("start mark3")
        # build inital map
        jobnames = self.job_info_dict.keys()
        data = np.zeros((len(jobnames), len(sys_nodes)), dtype=int) # initial fake data
        initialMap = pd.DataFrame(data=data, index=jobnames, columns=sys_nodes)
        
        print("start mark4")

        # Feed the optimizer and get the optimized newmap
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        print("start mark5")
        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=initialMap.values, jmin=mins, jmax=maxs,
                                                            Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, 
                                                            time_limit=10)
        print(new_data)

        new_map = pd.DataFrame(data=new_data, index=jobnames, columns=sys_nodes)

        print("start mark6")
        managerOperations.adjust_nodes_by_map(new_map=new_map, old_map=initialMap, job_info_dict = self.job_info_dict)
        print("start mark7")
        self.current_map = new_map

    def scheduler_job_change(self, GUIDs):
        # 1.job leave
        for GUID in GUIDs:
            # 1.1 Drop job from current map
            self.current_map.drop([GUID], inplace=True)
            # 1.2 Delete jobInfo from jobInfo dictionary
            self.job_info_dict.pop(GUID)

        # 2. fetch jobs to max parallel(batch manner)
        job_string_list = []
        lacking_len = self.max_parallel - len(self.job_info_dict)
        job_num = min(lacking_len, self.get_job_queue_len())
        for i in range(job_num):
            jobstring = self._get_a_job_from_DB()
            job_string_list.append(jobstring)
            jobdetail = utils.parser_job_string_2_job_item(jobstring)
            newrow = pd.DataFrame([np.zeros(len(self.current_map.columns), dtype=int)], index=[jobdetail.GUID] ,columns=self.current_map.columns)
            # add new job to map
            self.current_map = self.current_map.append(newrow)
            # add new job to dict
            self.job_info_dict[jobdetail.GUID] = jobdetail

        # get parameters with latest `job_info_dict`
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map.values, jmin=mins, jmax=maxs,
                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)

        new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)

        managerOperations.adjust_nodes_by_map(new_map, self.current_map, self.job_info_dict)

        # update current_map
        self.current_map = new_map

    def scheduler_nodes_change(self, flag, nodes):
        # drop or add columns for nodes in cmap
        old_map = self.current_map
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        if flag == JobNodeStatus.NODEIN:
            for node in nodes:
               self.current_map.insert(self.current_map.shape[1], node, 0) # dataframe add one new column
            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map.values, jmin=mins, jmax=maxs,
                                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)
            new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)
        else:
            for node in nodes:
                tmp_map = self.current_map.drop(labels=node, axis=1) # dataframe delete one column

            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=tmp_map.values, jmin=mins, jmax=maxs,
                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)
            new_map = pd.DataFrame(data=new_data, index=tmp_map.index, columns=tmp_map.columns)
        
        managerOperations.adjust_nodes_by_map(new_map, old_map, self.job_info_dict)

        # update current_map
        self.current_map = new_map

    def _terminate_manager(self):
        # kill all horovodrun process in dictionary
        for job in self.job_info_dict:
            p = psutil.Process(self.job_info_dict[job].pid) 
            p.terminate() # kill all the running process for this job

        # clean all host file and discovery file
        for job in self.job_info_dict:
            managerOperations.del_host_files(jobname=job)
    
    # TODO: change the way triggering re-allocation
    def _monitor_hvd_processes(self):
        while True:
            print("============= monitor report ===============")
            sleep(self.monitor_gap) # check every 15s
            for jobname in self.job_info_dict.keys():
                pid = self.job_info_dict[jobname].pid
                if not psutil.pid_exists(pid):
                    self.scheduler_job_change(GUIDs=jobname)
            print("============= monitor report ===============")
    
    def merge_ordered_NO_2_itemNO(self, job_N, job_O, N, O):
        print("====== Merge ordered NO 2 itemNO called ========")
        # Switch job_N and job_O to a dict

        print('print four items: ', job_N, job_O, N, O)

        item_no_dict = {}
        for i in range(len(job_N)):
            print("key job Ni", job_N[i])
            print("val job Oi", job_O[i])
            item_no_dict[job_N[i]] = job_O[i]

        # insert N and O to the tmp dict
        # Key same will replace, otherwise will insert
        for i in range(len(N)):
            print("insert key Ni", N[i])
            print("insert val Oi", O[i])
            item_no_dict[N[i]] = O[i]

        # Sort the merged dict
        ordered_job_dict = OrderedDict(sorted(item_no_dict.items()))

        print("ordered job dict", ordered_job_dict)

        # convert back to N and O (ordered)
        res_N = list(ordered_job_dict.keys())
        res_O = list(ordered_job_dict.values())

        # print res_N and res_O
        print("res_N", res_N)
        print("res_O", res_O)

        return res_N, res_O

    def dynamic_update_job_data(self, jobname, N=None, O=None, res_up = None, res_down = None):
        """
        This function only for dynamic update job_info_dict job N O and resup and down information
        """
        print("========Dynamic update job data called==========")
        print("N is: ", N)
        print("O is: ", O)

        if self.job_info_dict == None or len(self.job_info_dict) == 0:
            return

        if jobname == None or len(jobname) == 0:
            return

        job_item = self.job_info_dict[jobname] # assume we can find this job info by name

        if N != None or O != None:
            if job_item.N == None or job_item.O == None:
                job_item.N = N
                job_item.O = O
            else:
                # if not None then get the merged N and O
                res_N, res_O = self.merge_ordered_NO_2_itemNO(job_N=job_item.N, job_O=job_item.O, N=N, O=O)

                # update job item
                job_item.N = res_N
                job_item.O = res_O
        
        if res_up != None:
            job_item.res_up = res_up

        if res_down != None:
            job_item.res_down = res_down

    def update_job_data(self, mserver):
        print("start update data")
        while True:
            sleep(20) # pin gap
            print("update data every 10 seconds")
            msg_items = []
            msg_list = mserver.buffer

            if len(msg_list) == 0:
                continue

            for msg in msg_list:
                msg_item = utils.parser_udp_message(msg)
                msg_items.append(msg_item)

            # group by address
            # ??? every time we sort and group by (is this is necessary??)
            msg_items.sort(key=lambda x: x.address)
            group_items = groupby(msg_items, lambda x: x.address)
            
            group_dict = {} # key:job val:group of iterations info for this job
            for key, group in group_items:
                hostname = utils.get_host_name_by_address(key)
                print("====hostname====:", hostname)
                jobname = utils.get_jobname_by_hostname(hostname, self.current_map)
                print("jobname", jobname)
                group_dict[jobname] = list(group)

            print("group dict", group_dict)

            # get N and O
            for jobname in group_dict:
                job_items = group_dict[jobname]
                job_items.sort(key=lambda x: x.rank_size)
                group_job_items = groupby(job_items, lambda x: x.rank_size)
                
                print(group_job_items)

                N = []
                O = []
                for key, group in group_job_items: # key: different rank size for job - group: items of this job with this ranksize
                    node_num = int(key)/NUM_OF_GPUs_PER_NODE
                    N.append(int(node_num))
                    group_list = list(group)
                    avg = sum([float(x.credit) for x in group_list])/len(group_list)
                    O.append(avg)

                print(N)
                print(O)

                # get res_up and res_down
                job_items.sort(key=lambda x: x.id)
                res_up = None
                res_dw = None
                if len(job_items) > 2:
                    print("update res up and down")
                    for i in range(len(job_items)-1,-1,-1): # reverse order find rank difference
                        if job_items[i].rank_size > job_items[i-1].rank_size and job_items[i-1].rank_size == job_items[i-2].rank_size:
                            res_up = (job_items[i].time - job_items[i-1].time) - (job_items[i-1].time - job_items[i-2].time)
                        elif job_items[i].rank_size < job_items[i-1].rank_size:
                            res_dw = (job_items[i].time - job_items[i-1].time) - (job_items[i-1].time - job_items[i-2].time)
                print("res_up",res_up)
                print("res_dw",res_dw)

            self.dynamic_update_job_data(jobname=jobname, N=N, O=O, res_up=res_up, res_down=res_dw)

    def run_server_and_update_data(self):
        # run server daemon
        print("run server and update data")
        mserver=MSGOperations()
        p_server = Thread(target=mserver.create_udp_server)
        p_server.start()

        print("start run update job data")
        # run update daemon
        p_updater = Thread(target=self.update_job_data, args=(mserver,))
        p_updater.start()

def main():
    # create manager
    m = Manager(max_parallel=MAXIMUM_PARALLEL, monitor_gap= 10)

    # run udp server and update job data
    m.run_server_and_update_data()

    # start jobs
    print("before manager start")
    m._managerStart()
    print("after manager start")

    # node leave
    # sleep(40)
    # print("node leave")
    # m.scheduler_nodes_change(JobNodeStatus.NODEOUT, ["thetagpu21"])
    
    # node in
    # m.scheduler_nodes_change(JobNodeStatus.NODEIN, ["node10"])

if __name__ == "__main__":
    main()
