from itertools import groupby
from queue import Queue
import time
import sys

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
import trace_generator

MAXIMUM_PARALLEL = 2 # 

MONITOR_GAP = 20
NUM_OF_GPUs_PER_NODE = 8 #

class JobNodeStatus(Enum):
    JOBFAILED=0
    JOBOUT=1
    NODEIN=2
    NODEOUT=3

class Manager:
    def __init__(self, max_parallel = MAXIMUM_PARALLEL, monitor_gap = MONITOR_GAP):
        self.max_parallel = max_parallel
        self.monitor_gap = monitor_gap
        self.create_working_directory()
        self.buffer = Queue()

        # create server ready to recv data
        self.run_msg_server()

    current_map = pd.DataFrame()
    # A dictionary for global job information
    job_info_dict = {}

    # create working directory
    def create_working_directory(self):
        managerOperations.create_working_directory()

    # Network Operations
    def create_msg_client(self, address, port):
        """Create a message client by client for reporting training throughput/speed.

        Args:
            func (func pointer): Client need to offer a function to get the real time training speed
        """
        return MSGOperations().create_msg_client(address, port)

    def create_msg_server(self): # pass dynamic update data function into 
        """Create a message server for receive throughput report by client

        Args:
            func (function pointer): a function for deciding when to trigger update job data
        """
        managerOperations.create_msg_server()

    # life cycle functions
    def manager_start(self):
        sys_nodes=sys_admin.get_cluster_nodes()
        if len(sys_nodes) == 0 or (sys_nodes is None):
            return
        
        if utils.get_job_queue_len() == 0:
            return
        
        # fetch job from DB
        starting_jobs_num = min(MAXIMUM_PARALLEL, utils.get_job_queue_len())
        for i in range(starting_jobs_num):
            job_string = utils.get_a_job_from_DB()
            jobdetail = utils.parser_job_string_2_job_item(job_string)
            self.job_info_dict[jobdetail.GUID] = jobdetail
        
        # build inital map
        jobnames = self.job_info_dict.keys()
        data = np.zeros((len(jobnames), len(sys_nodes)), dtype=int) # initial fake data
        initialMap = pd.DataFrame(data=data, index=jobnames, columns=sys_nodes)

        # Feed the optimizer and get the optimized newmap
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=initialMap, jmin=mins, jmax=maxs,
                                                            Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, 
                                                            time_limit=10)
        print(new_data)

        new_map = pd.DataFrame(data=new_data, index=jobnames, columns=sys_nodes)

        managerOperations.adjust_nodes_by_map(new_map=new_map, old_map=initialMap, job_info_dict = self.job_info_dict)
        self.current_map = new_map
        
        print("=======================")
        print("===manager start end===")
        print("=======================")

    def scheduler_job_change(self, GUIDs):
                
        # 1. Detect job leave
        # drop the job from map and job_dict
        for GUID in GUIDs:
            print("GUID: ", GUID)
            print("GUID Type: ", type(GUID))
            # 1.1 Drop job from current map
            self.current_map.drop([GUID], inplace=True)
            # 1.2 Delete jobInfo from jobInfo dictionary
            self.job_info_dict.pop(GUID)

        # 2. fetch jobs to max parallel(batch manner)
        print("job leaving after and fetch before job_info_dict: ", self.job_info_dict)
        print("fetching job to max parallel")

        job_string_list = []
        lacking_len = self.max_parallel - len(self.job_info_dict)
        print(self.max_parallel)
        print("lacking_len: ", lacking_len)

        left_jobs_in_db = utils.get_job_queue_len()
        print("There are %d jobs left in database" % left_jobs_in_db)

        job_num = min(lacking_len, left_jobs_in_db)
        print("fetched valid job number:", job_num)

        for i in range(job_num):
            jobstring = utils.get_a_job_from_DB()
            print("fetch new job: ", jobstring)
            job_string_list.append(jobstring)
            jobdetail = utils.parser_job_string_2_job_item(jobstring)
            newrow = pd.DataFrame([np.zeros(len(self.current_map.columns), dtype=int)], index=[jobdetail.GUID] ,columns=self.current_map.columns)
            
            # add new job to map
            self.current_map = self.current_map.append(newrow)
            # add new job to dict
            self.job_info_dict[jobdetail.GUID] = jobdetail

        print("After fetching jobs get the map for adjustment")
        print(self.current_map)

        # update buffer info to job_info_dict
        # 
        self.update_job_data_on_events(self.buffer, event_type="job event")
        
        # get parameters with latest `job_info_dict`
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map, jmin=mins, jmax=maxs,
                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)

        new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)

        managerOperations.adjust_nodes_by_map(new_map, self.current_map, self.job_info_dict)

        # update current_map
        self.current_map = new_map
        
        # print current map after allocation
        print("After re-allocation jobs get the new map")
        print(self.current_map)

    def scheduler_nodes_change(self, flag, nodes):
        print("node change called")

        # Event driven update data here
        # Use buffer data and get ns os and res_up and res_dw
        # 
        self.update_job_data_on_events(self.buffer, event_type= "node event")

        # validate nodes name before operations 
        if sys_admin.is_nodes_belong_to_avaliable_nodes(nodes) == False:
            print("error: nodes out of avaliable nodes range")
            return

        # drop or add columns for nodes in cmap
        old_map = self.current_map
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        print("Ns Os res_ups res_dw")
        print(Ns, Os, res_ups, res_dws)

        if flag == JobNodeStatus.NODEIN:
            print("node in :", nodes)
            print("old map", self.current_map)
            for node in nodes:
               self.current_map.insert(self.current_map.shape[1], node, 0) # dataframe add one new column
            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map, jmin=mins, jmax=maxs,
                                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)
            new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)
        else:
            print("node leave ", nodes)
            print("old map", self.current_map)
            for node in nodes:
                tmp_map = self.current_map.drop(labels=node, axis=1) # dataframe delete one column

            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=tmp_map, jmin=mins, jmax=maxs,
                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10)
            new_map = pd.DataFrame(data=new_data, index=tmp_map.index, columns=tmp_map.columns)
            print("new map", new_map)

        managerOperations.adjust_nodes_by_map(new_map, old_map, self.job_info_dict)

        # update current_map
        self.current_map = new_map

    # for future usage
    def _terminate_manager(self):
        # kill all horovodrun process in dictionary
        for job in self.job_info_dict:
            p = psutil.Process(self.job_info_dict[job].pid) 
            p.terminate() # kill all the running process for this job

        # clean all host file and discovery file
        for job in self.job_info_dict:
            managerOperations.del_host_files(jobname=job)
    
    def monitor_hvd_processes(self):
        while True:
            now = time.time()
            print("============= %f : monitor hvd process report * Head ===============" % now)
            
            time.sleep(self.monitor_gap)
            jobnames = []
            print(self.current_map)
            for jobname in self.job_info_dict.keys():
                pid = self.job_info_dict[jobname].pid
                print("dict stored hvd job - pid: %s - %s" % (jobname, pid))

                if not psutil.pid_exists(pid):

                    print("Hey! we found hvdrun job: %s - process: %s not existed anymore, so we start fetch new jobs" % (jobname, pid))
                    jobnames.append(jobname)

            if jobnames:
                print("jobnames", jobnames)
                self.scheduler_job_change(GUIDs=jobnames)

            print("============= %f : monitor hvd process report * foot ===============" % now)
    
    def merge_ordered_NO_2_itemNO(self, job_N, job_O, N, O):
        # Switch job_N and job_O to a dict
        item_no_dict = {}
        for i in range(len(job_N)):
            item_no_dict[job_N[i]] = job_O[i]

        # insert N and O to the tmp dict
        # Key same will replace, otherwise will insert
        for i in range(len(N)):
            print("insert key Ni", N[i])
            print("insert val Oi", O[i])
            item_no_dict[N[i]] = O[i]

        # Sort the merged dict
        ordered_job_dict = OrderedDict(sorted(item_no_dict.items()))

        # convert back to N and O (ordered)
        Ns = list(ordered_job_dict.keys())
        Os = list(ordered_job_dict.values())

        # print res_N and res_O
        print("Ns: ", Ns)
        print("Os: ", Os)

        return Ns, Os

    def update_scaling_and_cost_data_2_jobInfoDict(self, jobname, N=None, O=None, res_up = None, res_down = None):
        """
        This function only for dynamic update job_info_dict job N O and resup and down information
        """

        print("========update_scaling_and_cost_data_2_jobInfoDict called==========")
        print("N is: ", N)
        print("O is: ", O)
        print("res_up is", res_up)
        print("res_dw is", res_down)
        
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

    def update_job_data_on_events(self, buffer, event_type):
        ''' job change or node change trigger updating data for re-allocation'''

        if len(buffer) == 0:
            print("No valid information and skip update process")
            return

        print("buffer len:",len(buffer))

        for key in buffer:
            print("buffer item len:", len(buffer[key]))

        # get job msg_item dict
        group_dict = {}
        for key in buffer:
            hostname = utils.get_host_name_by_address(key)
            jobname = utils.get_jobname_by_hostname(hostname, self.current_map)

            if jobname == "":
                continue
            
            msg_list = buffer[key]
            msg_items = []
            for msg in msg_list:
                msg_item = utils.parser_udp_message(msg)
                print("======msg information=======")
                print(msg_item.address)
                print(msg_item.credit)
                msg_items.append(msg_item)
    
            group_dict[jobname] = msg_items

        # Cannot find job related
        if len(group_dict) == 0:
            print("Jobs in current_map has no any training msg in buffer, probably means all jobs are new fetched(not started yet)," \
            " do not need to use historial information to do the update, so return the update function here")
            return

        # for each job sort by id or rank ect.
        for jobname in group_dict:
            job_items = group_dict[jobname]
            job_items.sort(key=lambda x: x.id) # sorted by id
            job_items.sort(key=lambda x: x.rank_size)
            group_job_items = groupby(job_items, lambda x: x.rank_size)

            N = []
            O = []
            for key, group in group_job_items: # key: different rank size for job - group: items of this job with this ranksize
                node_num = int(key)/NUM_OF_GPUs_PER_NODE
                N.append(int(node_num))
                group_list = list(group)
                
                thrputs = []
                for i in range(0, len(group_list) - 1):
                    msg_time_gap = float(group_list[i + 1].time) - float(group_list[i].time)
                    thrput = float(group_list[i].credit) / msg_time_gap
                    thrputs.append(thrput)
                avg_thrput = thrputs[-1] # use the last thrput as the current thrput

                O.append(avg_thrput)

            # get res_up and res_down
            job_items.sort(key=lambda x: x.id)
            res_up = None
            res_dw = None
            if len(job_items) > 2:
                print("update res_up and res_dw")

                for i in range(len(job_items)-1,-1,-1): # reverse order find rank difference
                    '''
                    print("i rank_size:", job_items[i].rank_size)
                    print("i- 1 rank_size:", job_items[i-1].rank_size)
                    print("i - 2 rank_size:", job_items[i-2].rank_size)
                    print(type(job_items[i-2].rank_size))
                    print("idx value: ", i)
                    '''
                    
                    if job_items[i].rank_size > job_items[i-1].rank_size and job_items[i-1].rank_size == job_items[i-2].rank_size:
                        print("================ get the res_up cost ===================")
                        res_up = (job_items[i].time - job_items[i-1].time) - (job_items[i-1].time - job_items[i-2].time)
                        print("==resup==", res_up)

                    elif job_items[i].rank_size < job_items[i-1].rank_size and job_items[i-1].rank_size == job_items[i-2].rank_size:
                        print("================ get the res_down cost =================")
                        print(type(job_items[i].time))
                        print(job_items[i].time)
                        res_dw = (job_items[i].time - job_items[i-1].time) - (job_items[i-1].time - job_items[i-2].time)
                        print("==resdw==", res_dw)

            print("res_up", res_up)
            print("res_dw", res_dw)
        
        # Update collect info to JobInfoDict
        self.update_scaling_and_cost_data_2_jobInfoDict(jobname=jobname, N=N, O=O, res_up=res_up, res_down=res_dw)

    def run_msg_server(self):
        # run server daemon
        print("run server and update data")
        mserver=MSGOperations()
        self.buffer = mserver.buffer

        p_server = Thread(target=mserver.create_msg_server) # keep updating buffer data
        p_server.start()

    def is_first_round(self, counters):
        flag = True
        for coun in list(counters.values()):
            if coun != 0:
                flag = False
        return flag

    # node come and leave
    def events_launcher(self):
        self.manager_start()

        # create events source
        cluster_nodes = sys_admin.get_cluster_nodes()
        trace = trace_generator.synthetic_trace(nodes=cluster_nodes, nf=20000)
        
        # write events trace to log
        with open("events.log", 'w') as f:
            f.write(str(trace))

        # record the status of nodes in arrary for controlling events      
        counters = {}
        flags = {}
        for node in cluster_nodes:
            counters[node] = 0
            flags[node] = False

        start_time = time.time()

        # events tick driver
        while(True):
            time.sleep(0.1) # check each 0.1 second(could change)
            
            passing_time = time.time() - start_time

            coming_nodes = [] # coming nodes in one time check
            leaving_nodes = [] # leaving nodes in one time check

            for node in cluster_nodes:
                timestamps_tuple = trace[node] # list of tuple for node
                current_time_tuple = timestamps_tuple[counters[node]]

                if passing_time > current_time_tuple[0] and flags[node] == False:
                    print("node: " + node + " in") # trigger node in event
                    coming_nodes.append(node)
                    flags[node] = True

                if passing_time > current_time_tuple[1] and flags[node] == True:
                    print("node: " + node + " leave") # trigger node leave event
                    leaving_nodes.append(node)
                    flags[node] = False
                    counters[node] = counters[node] + 1
            
            # when node counter == 0 that is the start, just skip the start phase
            if coming_nodes and self.is_first_round(counters) == False:
                self.scheduler_nodes_change(JobNodeStatus.NODEIN, coming_nodes)

            if leaving_nodes:
                self.scheduler_nodes_change(JobNodeStatus.NODEOUT, leaving_nodes)

def main():

    # create manager
    # basic settings and msg server
    m = Manager()

    # 2. start jobs and run as events come
    p_events = Thread(target=m.events_launcher)
    p_events.start()
    
    # 3. process monitor job pid
    p_monitor = Thread(target=m.monitor_hvd_processes)
    p_monitor.start()

    # for local testing purpose
    


if __name__ == "__main__":
    main()
