import os
import DBOperations
from msgOperations import MSGOperations
from subprocess import Popen
import utils
import psutil
import stat

NUM_OF_GPUs_PER_NODE = 1
WORKING_DIR = utils.working_dir()

def create_working_directory():
    work_dir = os.path.exists(WORKING_DIR)
    if not work_dir:
        os.mkdir(WORKING_DIR)

def submit_job(min, max, Ns, Os, res_up, res_dw, path):

    # node string
    node_range_str = "min:%d max:%d" % (min, max)

    # Ns and Os string
    ns_str = "Ns:" + ",".join([str(_) for _ in Ns])
    os_str = "Os:" + ",".join([str(_) for _ in Os])
    
    # res string
    resup_str = "res_up:" + str(res_up)
    resdown_str = "res_dw:" + str(res_dw)

    # horovod command string
    path_str = "path:" + path
    jobString = " ".join([node_range_str, ns_str, os_str, resup_str, resdown_str, path_str])

    return DBOperations.submit_job_2_DBQueue(jobString)

def get_job_queue_len():
    return DBOperations.get_DB_queue_len()

def get_a_job_from_DB():
    if get_job_queue_len() > 0:
        return DBOperations.get_Job_from_DBQueue()
    else:
        return None

def create_msg_client(address, port):
    return MSGOperations().create_udp_client(address, port)

def create_msg_server(): # pass dynamic update data function into 
    MSGOperations().create_udp_server()

def add_job(jobname, nodes, job_info_dict):
    print("Add job was called")
    # if the job not being assigned node
    # just skip the process
    if nodes == None or len(nodes) == 0:
        return

    # 1. Create discovery and host file
    discover_file_path = create_discovery_file(jobname, nodes)
    command = generate_command(discover_file_path, jobname, job_info_dict)
    
    # mkl service error
    # refer this issue https://github.com/pytorch/pytorch/issues/37377
    myenv = os.environ
    myenv["MKL_SERVICE_FORCE_INTEL"] = "1"

    # 2. Launch new job and get process id
    with open("stdout.txt","w") as out, open("stderr.txt","w") as err:
        print("before start new job")
        print("command", command)
        p = Popen(command, shell=True, env=myenv, stdout=out, stderr=err)
        print("after start new job")

        hvdrunParentPid = p.pid
        fp = psutil.Process(hvdrunParentPid)

        hvdpid = fp.children()[0].pid
        print("hvdpid is:", hvdpid)
        # 3. update process id to `jobInfoDict`
        jobItem = job_info_dict[jobname]
        jobItem.pid = hvdpid

def create_discovery_file(jobname, nodes):
    discovery_path = os.path.join(WORKING_DIR,"discover_host_" + jobname + ".sh")
    with open(discovery_path, 'w') as w:
        w.write("#!/bin/bash\n")
        w.write("echo node06:0\n")
        for node in nodes:
            w.write("echo " + node + ":" + str(NUM_OF_GPUs_PER_NODE) + "\n")

    # grant host file executable permission
    st = os.stat(discovery_path)
    os.chmod(discovery_path, st.st_mode | stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
    return discovery_path

def generate_command(discover_file_path, jobname, job_info_dict):
    scriptPath = job_info_dict[jobname].path
    command = "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/horovodrun -np 1 --host-discovery-script " + discover_file_path + " python " + scriptPath
    return command

def del_job(jobname, job_info_dict):
    # 1. check the pid is running
    job_pid = job_info_dict[jobname].pid
    if job_pid == -1:
        del_discover_files(jobname)
        return
    
    # 2. kill process
    if psutil.pid_exists(job_pid):
        p = psutil.Process(job_pid)
        p.terminate()

    # 3. remove job from dict
    if job_info_dict.has_key(jobname):
        job_info_dict.pop(jobname)

    # 3. remove discover and host files
    del_discover_files(jobname)

def del_discover_files(jobname):
    discovery_file_path = os.path.join(WORKING_DIR, "discover_host_" + jobname + ".sh")

    if os.path.exists(discovery_file_path):
        os.remove(discovery_file_path)

# Node changes
def add_nodes_for_job(jobname, nodes):
    if len(nodes) == 0:
        return

    host_file_path = os.path.join(WORKING_DIR, jobname + "_hostfile") 
    if os.path.exists(host_file_path):
        with open(host_file_path, 'a') as w:
            for node in nodes:
                w.write(node + ":" + str(NUM_OF_GPUs_PER_NODE)  + "\n")

def del_nodes_for_job(jobname, nodes):
    # del host from corresponding hostfile
    host_file_path = os.path.join(WORKING_DIR, jobname + "_hostfile")
    if os.path.exists(host_file_path):
        lines = []
        with open(host_file_path, 'r') as r:
            lines = r.readlines()
        
        new_lines = []
        for node in nodes:
            for line in lines:
                if node not in line:
                    new_lines.append(line)

        with open(host_file_path, 'a') as w:
            for line in new_lines:
                w.write(line)

def adjust_nodes_by_map(new_map, old_map, job_info_dict):
    """Make adjustment by comparing two maps differences

    Args:
        newMap (dataframe): the input dataframe from optimizer
    """
    # map to dict
    old_job_nodes_dict = utils.get_job_nodes_mapping_from(old_map)
    new_job_nodes_dict = utils.get_job_nodes_mapping_from(new_map)
    
    print("old dict",old_job_nodes_dict)
    print("new dict",new_job_nodes_dict)
    
    # Adjustment on job level
    oldjobs = list(old_job_nodes_dict.keys())
    newjobs = list(new_job_nodes_dict.keys())

    oldjobs.sort()
    newjobs.sort()

    if oldjobs != newjobs:
        for oldjob in oldjobs:
            if oldjob not in newjobs:
                del_job(oldjob) 
        
        for newjob in newjobs:
            if newjob not in oldjobs:
                print("mark 1 add job before")
                add_job(newjob, new_job_nodes_dict[newjob], job_info_dict)
                print("makr 1 add job after")
    
    # Adjustment on node level
    overlappedJobs = utils.get_lists_overlap(newjobs, oldjobs)

    # for each job check node changes
    for job in overlappedJobs:
        old_nodes = old_job_nodes_dict[job]
        new_nodes = new_job_nodes_dict[job]
        # nothing changed skip
        if set(old_nodes) != set(new_nodes):
            # pid = -1 means this job never launched before
            # len(old_nodes) == 0 means the no node assigned for this job
            if job_info_dict[job].pid == -1 and len(old_nodes) == 0:
                add_job(job, new_job_nodes_dict[job], job_info_dict)
                continue
            
            # job existed adjust nodes only
            intersectionNodes = utils.get_lists_overlap(old_nodes, new_nodes)
            addNodes =list(set(new_nodes) - set(intersectionNodes))
            add_nodes_for_job(jobname=job, nodes=addNodes)
            delNodes = list(set(old_nodes) - set(intersectionNodes))
            del_nodes_for_job(jobname=job, nodes=delNodes)
