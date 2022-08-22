from jobInfo import JobInfo
import os
import socket
import DBOperations

CSTART = '\033[1;31;40m'
CEND = '\033[0m'

class UDP_Msg:
    def __init__(self, address, id, time, rank_size, credit, jobname):
        self.address = address
        self.id = id
        self.time = time
        self.rank_size = rank_size
        self.credit = credit
        self.jobname = jobname

def get_lists_overlap(nums1, nums2):
    nums1.sort()
    nums2.sort()
    length1, length2 = len(nums1), len(nums2)
    intersection = list()
    index1 = index2 = 0
    while index1 < length1 and index2 < length2:
        num1 = nums1[index1]
        num2 = nums2[index2]
        if num1 == num2:
            if not intersection or num1 != intersection[-1]:
                intersection.append(num1)
            index1 += 1
            index2 += 1
        elif num1 < num2:
            index1 += 1
        else:
            index2 += 1
    return intersection

def parser_job_string_2_job_item(jobString):
    """
    
    jobstring example for reference
    name:job1 min:1 max:6 N:1,2,3 O:1,1.9,2.8 res_up:3 res_down:1 path:train.py

    jobstring example for reference
    name:job1 min:1 max:6 N:1,2,3 O:1,1.9,2.8 res_up:3 res_down:1 path:train.py --jobaname 1

    """

    # Get job dict from string
    jobString = jobString.split(" --")[0] # delete when do not need to report jobname

    jobDict = {}
    items = jobString.split(" ")
    for item in items:
        key = item.split(":")[0]
        if key == "N": # get Nodes val(ordered)
            val = list(int(x) for x in item.split(":")[1].split(","))
            val.sort()
        elif key == "O": # get Objective val(ordered)
            val = list(float(x) for x in item.split(":")[1].split(","))
            val.sort()
        else:
            val = item.split(":")[1]

        jobDict[key] = val

    jobdetail = JobInfo(GUID=jobDict["GUID"],
                        pid=-1,                 # default set pid -1
                        max=jobDict["max"], 
                        min=jobDict["min"], 
                        N=jobDict["N"], 
                        O=jobDict["O"], 
                        resUp=jobDict["res_up"],
                        resDown=jobDict["res_dw"],
                        path=jobDict["path"]
                        )
    return jobdetail

def get_optimizer_parameters_by_job_dict(jobInfoDict):
    jobnames, mins, maxs, Ns, Os, res_ups, res_dws= [], [], [], [], [], [], []

    for Guid in jobInfoDict.keys():
        jobdetail = jobInfoDict[Guid]
        jobnames.append(jobdetail.GUID)
        mins.append(int(jobdetail.min))
        maxs.append(int(jobdetail.max))
        Ns.append(jobdetail.N)
        Os.append(jobdetail.O)
        res_ups.append(float(jobdetail.resUp)) # need to profile upon specific job
        res_dws.append(float(jobdetail.resDown)) # need to profile upon specific job

    return mins, maxs, Ns, Os, res_ups, res_dws

def get_job_nodes_mapping_from(cmap):
    """convert map to dictionary

    Args:
        map (dataframe): a dataframe contains status information

    Returns:
        Dict: a dictionary converted by dataframe status
        The key is job name
        The value is the job corresponding nodes
    """
    cols = list(cmap.columns)
    indexs = list(cmap.index)
    dict = {}
    for job in indexs:
        nodes = []
        for node in cols:
            if int(cmap.at[job, node]) == 1:
                nodes.append(node)
        dict[job] = nodes

    return dict

def get_jobname_by_hostname(hostname, cmap):
    print("get_jobname_by_host_name() called")
    jobnodesdict = get_job_nodes_mapping_from(cmap)
    
    jobname = ""
    for jn in jobnodesdict:
        nodes = jobnodesdict[jn]
        if hostname in nodes:
            jobname = jn
            break
    return jobname

def parser_udp_message(msg):
    if msg == None or len(msg) == 0:
        return
    items = msg.split(" ")
    address = items[0].split(":")[-1]
    id = items[1].split(":")[-1]
    time = float(items[2].split(":")[-1])
    rank_size = int(items[3].split(":")[-1])
    credit = float(items[4].split(":")[-1])
    jobname = items[5].split(":")[-1][0:-1] # The last [0:-1] to remove the \n in the end of the string

    return UDP_Msg(address, id, time, rank_size, credit, jobname)

def working_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".BFTrainer")

def DB_path():
    return os.path.join(working_dir(), "jobdata.db")

def is_theta_cluster():
    if socket.gethostname().startswith("theta"):
        return True
    return False

def get_host_name_by_address(address):
    host_tuple = socket.gethostbyaddr(address)
    # this resolve is specific for thetagpu cluster
    hostname = host_tuple[0].split(".")[0]
    return hostname

## DB Operations
def submit_job(min, max, N, O, res_up, res_dw, path):

    # node string
    node_range_str = "min:%d max:%d" % (min, max)

    # N and O string
    n_str = "N:" + ",".join([str(_) for _ in N])
    o_str = "O:" + ",".join([str(_) for _ in O])
    
    # res string
    resup_str = "res_up:" + str(res_up)
    resdown_str = "res_dw:" + str(res_dw)

    # horovod command string
    path_str = "path:" + path
    jobString = " ".join([node_range_str, n_str, o_str, resup_str, resdown_str, path_str])

    return DBOperations.submit_job_2_DBQueue(DB_path(), jobString)

def get_job_queue_len():
    return DBOperations.get_DB_queue_len(DB_path())

def get_a_job_from_DB():
    if get_job_queue_len() > 0:
        return DBOperations.get_Job_from_DBQueue(DB_path())
    else:
        return None

def print_red(log_str):
    print(f"{CSTART}{log_str}{CEND}")

def print_colored_log(input, color = "RED"):
    if color == "RED":
        cstart = "\033[91m"
    elif color == "GREEN":
        cstart = "\033[92m"
    elif color == "YELLOW":
        cstart = "\033[93m"
    elif color == "BLUE":
        cstart = "\033[94m"
    elif color == "PURPLE":
        cstart = "\033[95m"

    print(f"{cstart}{input}{CEND}")
