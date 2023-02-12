import time
import pandas as pd
from enum import Enum
from collections import OrderedDict
import utils
from progMIP import re_allocate
import psutil
import numpy as np
import managerOperations
from msgOperations import MSGServer
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
        managerOperations.create_working_directory()
        self.current_map = pd.DataFrame() # map record the nodes allocation for current running jobs
        self.job_info_dict = {} # key:jobId value:jobInfo instance (maintain all current running jobs)
        self.mserver = MSGServer() # msg server instance
        self.cluster_nodes = sys_admin.get_cluster_nodes() # the whole cluster nodes
        # msg server to recv data from each training client
        self.msg_server_run()

    # life cycle functions
    def manager_start(self):
        if len(self.cluster_nodes) == 0 or (self.cluster_nodes is None):
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
        data = np.zeros((len(jobnames), len(self.cluster_nodes)), dtype=int) # initial fake data
        initialMap = pd.DataFrame(data=data, index=jobnames, columns=self.cluster_nodes)

        # Feed the optimizer and get the optimized newmap
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=initialMap, jmin=mins, jmax=maxs,
                                                            Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, 
                                                            time_limit=10,note="manager_start")

        new_map = pd.DataFrame(data=new_data, index=jobnames, columns=self.cluster_nodes)

        managerOperations.adjust_nodes_by_map(new_map=new_map, old_map=initialMap, job_info_dict = self.job_info_dict)
        self.current_map = new_map
        
        print("=======================")
        print("===manager start end===")
        print("=======================")

    def event_job_change(self, GUIDs):

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

        # get parameters with latest `job_info_dict`
        self.get_previous_running_jobs_data_and_update_jobInfoDict()
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map, jmin=mins, jmax=maxs,
                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10, note="job_change_event")

        new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)

        managerOperations.adjust_nodes_by_map(new_map, self.current_map, self.job_info_dict)

        self.current_map = new_map

    def event_nodes_change(self, flag, nodes, passing_time):
        utils.print_colored_log("A new node change event start(event node change func called)", color="PURPLE")

        # Validate nodes names
        if sys_admin.is_nodes_belong_to_avaliable_nodes(nodes) == False:
            print("error: nodes out of avaliable nodes range")
            return

        # Get job scale and update the info to jobinfoDict
        self.get_previous_running_jobs_data_and_update_jobInfoDict()

        # get the needed params for optimization
        mins, maxs, Ns, Os, res_ups, res_dws = utils.get_optimizer_parameters_by_job_dict(self.job_info_dict)

        utils.print_red("Finish the Optimization Parameters Setting before the optimization")
        print(f"Ns:{Ns} Os:{Os} res_ups:{res_ups} res_dw:{res_dws}")

        old_map = self.current_map.copy() # Deep copy current map here
        print("old map", self.current_map)

        if flag == JobNodeStatus.NODEIN:
            utils.print_red(f"node in:{nodes} before re_allocate()")
            for node in nodes:
               self.current_map.insert(self.current_map.shape[1], node, 0) # dataframe add one new column
            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=self.current_map, jmin=mins, jmax=maxs,
                                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10, note="node_change_nodein:" + passing_time)
            new_map = pd.DataFrame(data=new_data, index=self.current_map.index, columns=self.current_map.columns)
        else:
            utils.print_red(f"node leave:{nodes} before re_allocate()")
            for node in nodes:
                tmp_map = self.current_map.drop(labels=node, axis=1) # dataframe delete one column
            tmpGRB, new_data, tmpRate, tmpCost = re_allocate(cmap=tmp_map, jmin=mins, jmax=maxs,
                                                Ns=Ns,Os=Os, Tfwd=10, res_up=res_ups, res_dw = res_dws, time_limit=10, note= "node_change_nodeout:" + passing_time)
            new_map = pd.DataFrame(data=new_data, index=tmp_map.index, columns=tmp_map.columns)

        print("new map", new_map)

        managerOperations.adjust_nodes_by_map(new_map, old_map, self.job_info_dict)
        self.current_map = new_map

        utils.print_colored_log("=======Events node change on map was end==========", color="BLUE")

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
            print(f"============= {now} : monitor hvd process report * Head report every {MONITOR_GAP}s ===============")
            
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
                self.event_job_change(GUIDs=jobnames)

            print(f"============= {now} : monitor hvd process report * foot ===============")
    
    def merge_ordered_NO_2_itemNO(self, job_N, job_O, N, O):
        # Here job_N job_O N and O are all list
        # Switch job_N and job_O to a dict

        rank_obj_tmp_dict = {}
        for i in range(len(job_N)):
            rank_obj_tmp_dict[job_N[i]] = job_O[i]

        # insert N and O to the rank_obj_tmp_dict
        # Key same will replace(update), otherwise(rank not existed) will insert
        for i in range(len(N)):
            print("insert key Ni", N[i])
            print("insert val Oi", O[i])
            rank_obj_tmp_dict[N[i]] = O[i]

        # Sort the merged dict
        # TODO: no need to use OrderedDict
        ordered_job_dict = OrderedDict(sorted(rank_obj_tmp_dict.items()))

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
        
        if res_up != None and res_up != 0:
            job_item.res_up = res_up

        if res_down != None and res_down != 0:
            job_item.res_down = res_down

    # TODO: Working here
    def get_previous_running_jobs_data_and_update_jobInfoDict(self):
        ''' job change or node change trigger updating data for re-allocation'''
        
        if len(self.mserver.buffer) == 0:
            utils.print_colored_log("There is no any msg in buffer now", color="YELLOW")
        else:
            utils.print_colored_log(f"buffer keys is {self.mserver.buffer.keys()}", color="YELLOW")

        job_scale_info_dict = self.mserver.scale_info_dict

        print(f"mserver scale info dict keys {self.mserver.scale_info_dict.keys()}")

        # Co-exist job in 
        coexist_jobs_id = list(set(self.job_info_dict.keys()) & set(job_scale_info_dict.keys()))
        
        print(f"job info dict keys {self.job_info_dict.keys()}")
        print(f"job scale info dict keys {job_scale_info_dict.keys()}")
        print(f"Coexisted job ids {coexist_jobs_id}")

        for jobname in coexist_jobs_id:
            job_scale_info = job_scale_info_dict[jobname]
            nodes_nums = [rank/NUM_OF_GPUs_PER_NODE for rank in job_scale_info.rank_speed_dict.keys()]
            
            N = nodes_nums
            print(f"N list is {N}")

            O = list(job_scale_info.rank_speed_dict.values())
            print(f"O list is {O}")
            res_up = job_scale_info.add_overhead
            res_dw = job_scale_info.reduce_overhead

            # Update collect job scale info from server to JobInfoDict
            self.update_scaling_and_cost_data_2_jobInfoDict(jobname=jobname, N=N, O=O, res_up=res_up, res_down=res_dw)

    def msg_server_run(self):
        """create msg server and ready to monitor messages from clients
        """
        p_server = Thread(target=self.mserver.create_msg_server)
        p_server.start()

    def is_first_round(self, counters):
        flag = True
        for coun in list(counters.values()):
            if coun != 0:
                flag = False
        return flag

    def run_scheduler_events_simulator(self):
        """Simulate cluster nodes come and leave
        """
        # create events source
        trace = trace_generator.synthetic_trace(nodes=self.cluster_nodes, nf=20000)
        
        # write events trace to log
        with open("events.log", 'w') as f:
            f.write(str(trace))

        # record the status of nodes in arrary for controlling events      
        counters = {}
        flags = {}
        for node in self.cluster_nodes:
            counters[node] = 0
            flags[node] = False

        start_time = time.time()

        # Event loop
        while(True):
            time.sleep(0.1) # check each 0.1 second(could change)
            
            passing_time = time.time() - start_time

            coming_nodes = [] # coming nodes in one time check
            leaving_nodes = [] # leaving nodes in one time check

            for node in self.cluster_nodes:
                timestamps_tuple = trace[node] # list of tuple for node
                current_time_tuple = timestamps_tuple[counters[node]]

                if passing_time > current_time_tuple[0] and flags[node] == False:
                    utils.print_red("Event Simulator: Node " + node + " in at time:" + str(passing_time)) # trigger node in event
                    coming_nodes.append(node)
                    flags[node] = True

                if passing_time > current_time_tuple[1] and flags[node] == True:
                    utils.print_red("Event Simulator: Node: " + node + " leave at time:" + str(passing_time)) # trigger node leave event
                    leaving_nodes.append(node)
                    flags[node] = False
                    counters[node] = counters[node] + 1
            
            # when node counter == 0 that is the start, just skip the start phase
            if coming_nodes and self.is_first_round(counters) == False:
                self.event_nodes_change(JobNodeStatus.NODEIN, coming_nodes, str(passing_time) )

            if leaving_nodes:
                self.event_nodes_change(JobNodeStatus.NODEOUT, leaving_nodes, str(passing_time))

def main():

    # 1. Create manager
    # init manager and start running
    m = Manager()
    m.manager_start()

    # Run events simulator
    p_events = Thread(target=m.run_scheduler_events_simulator)
    p_events.start()
    
    # 3. process monitor job pid
    p_monitor = Thread(target=m.monitor_hvd_processes)
    p_monitor.start()

    # for local testing purpose
    
if __name__ == "__main__":
    main()
