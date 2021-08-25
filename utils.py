from jobInfo import JobInfo
import os
import socket

class UDP_Msg:
    def __init__(self, address, id, time, rank_size, credit):
        self.address = address
        self.id = id
        self.time = time
        self.rank_size = rank_size
        self.credit = credit

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
    """

    # Get job dict from string
    jobDict = {}
    items = jobString.split(" ")
    for item in items:
        key = item.split(":")[0]
        if key == "N":
            val = list(int(x) for x in item.split(":")[1].split(",")) 
        elif key == "O":
            val = list(float(x) for x in item.split(":")[1].split(",")) 
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

    print("&&&& Ns &&&&&", Ns)
    print("&&&& Os &&&&&", Os)

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
    jobnodesdict = get_job_nodes_mapping_from(cmap)
    for jn in jobnodesdict:
        nodes = jobnodesdict[jn]
        if hostname in nodes:
            jobname = jn
            break
    return jobname

def parser_udp_message(msg):
    if msg == None or len(msg) == 0:
        return
    print(msg)
    items = msg.split(" ")
    address = items[0].split(":")[-1]
    id = items[1].split(":")[-1]
    time = items[2].split(":")[-1]
    rank_size = items[3].split(":")[-1]
    credit = items[4].split(":")[-1]

    return UDP_Msg(address, id, time, rank_size, credit)

def working_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".BFTrainer")

def is_theta_cluster():
    if socket.gethostname().startswith("theta"):
        return True
    return False
