from jobInfo import JobInfo
import os
import DBOperations
from msgOperations import MSGOperations
from subprocess import Popen
import utils
import psutil
import stat

NUM_OF_GPUs_PER_NODE = 4

def submit_job(min, max, Ns, Os, res_up, res_dw, path):

    # node string
    node_range_str = "min:%d max:%d" % (min, max)

    # Ns and Os string
    ns_str = "Ns:" + ",".join([str(_) for _ in Ns])
    os_str = "Os:" + ",".join([str(_) for _ in Os])
    
    # res string
    resup_str = "res_up:" + str(res_up)
    resdown_Str = "res_dw:" + str(res_dw)

    # horovod command string
    path_str = "path:" + path
    jobString = " ".join([node_range_str, ns_str, os_str, resup_str, resdown_Str, path_str])

    return DBOperations.submit_job_2_DBQueue(jobString)

def get_job_queue_len():
    return DBOperations.get_DB_queue_len()

def get_a_job_from_DB():
    if get_job_queue_len() > 0:
        return DBOperations.get_Job_from_DBQueue()
    else:
        print("DEUBG:empty DB error")
        return None

def create_msg_client(address, port):
    return MSGOperations().create_udp_client(address, port)

def create_msg_server(): # pass dynamic update data function into 
    MSGOperations().create_udp_server()

def add_job(jobname, nodes, job_info_dict):
    
    # if the job not being assigned node
    # just skip the process
    if nodes == None or len(nodes) == 0:
        return

    print("Add job called")
    # 1. Create discovery and host file
    discover_file = create_discovery_and_host_file(jobname, nodes)
    command = generate_command(discover_file, jobname, job_info_dict)
    
    print("horovod command is: " + command)

    # mkl service error
    myenv = os.environ
    myenv["MKL_SERVICE_FORCE_INTEL"] = "1"

    # 2. Launch new job and get process id

    with open("stdout.txt","w") as out, open("stderr.txt","w") as err:
        p = Popen(command, shell=True, env=myenv, stdout=out, stderr=err)

        print("DEBUG: Success launch new process")
        hvdrunParentPid = p.pid
        print("DEBUG: PID is: ", hvdrunParentPid)

        fp = psutil.Process(hvdrunParentPid)

        print("DEBUG: ", fp.children())
        hvdpid = fp.children()[0].pid

        print("DEBUG:HOROVOD_PID:" + str(hvdpid))

        # 3. update process id to `jobInfoDict`
        jobItem = job_info_dict[jobname]
        jobItem.pid = hvdpid

def create_discovery_and_host_file(jobname, nodes):
    '''Create horovod elastic run essential files'''
    host_file = create_host_file(jobname, nodes)
    discovery_file = create_discovery_file(jobname, host_file)
    return discovery_file

def create_host_file(jobname, nodes):
    file_name = jobname + "_hostfile"
    if os.path.exists(file_name):
        return
    
    print("create host files")
    print(nodes)

    with open(file_name, "w") as w:
        for node in nodes:
            w.write(node + ':' + str(NUM_OF_GPUs_PER_NODE) + '\n')
    return file_name

def create_discovery_file(jobname, hostfile):
    # Create discovery file
    file_name = "discover_host_" + jobname + ".sh"
    if os.path.exists(file_name):
        return
    
    with open(file_name, 'w') as w:
        w.write("#!/bin/bash\n")
        w.write("\n")
        w.write("while read line\n")
        w.write("do\n")
        w.write("echo $line\n")
        w.write("done < ./" + hostfile)
    
    # Give the host file permission
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)

    return file_name

def generate_command(discover_file, jobname, job_info_dict):
    scriptPath = job_info_dict[jobname].path
    tmpPath = "~/ANL/scheduler/"  # TODO：this could be resolved(Probably)
    command = "horovodrun -np 1 --host-discovery-script " + tmpPath + discover_file + " python " + scriptPath
    return command

# And delete the discovery and host file at the same time
def del_job(jobname, job_info_dict):
    # 1. check the pid is running
    job_pid = job_info_dict[jobname].pid
    if job_pid == -1:
        del_host_files(jobname) # just clean the job content
        return
    
    # 2. kill process
    if psutil.pid_exists(job_pid):
        p = psutil.Process(job_pid)
        p.terminate()
        print("a process with pid %d was killed" % job_pid)
    else:
        print("a process with pid %d does not exist" % job_pid)

    # 3. remove job from dict
    if job_info_dict.has_key(jobname):
        job_info_dict.pop(jobname)

    # 3. remove discover and host files
    del_host_files(jobname)
    print("Job completely deleted")

def del_host_files(jobname):
    discovery_file = "discover_host_" + jobname + ".sh"
    host_file = jobname + "_hostfile"

    if os.path.exists(discovery_file):
        os.remove(discovery_file)

    if os.path.exists(host_file):
        os.remove(host_file)

# Node changes
def add_nodes_for_job(jobname, nodes):
    print("Call add nodes for job")
    if len(nodes) == 0:
        return
    # 1. Add host to corresponding hostfile
    host_file = jobname + "_hostfile"
    if os.path.exists(host_file):
        with open(host_file, 'a') as w:
            for node in nodes:
                w.write(node + ":" + str(NUM_OF_GPUs_PER_NODE)  + "\n")

def del_nodes_for_job(jobname, nodes):
    # del host from corresponding hostfile
    print("Call delete nodes for job")
    host_file = jobname + "_hostfile"
    if os.path.exists(host_file):
        lines = []
        with open(host_file, 'r') as r:
            lines = r.readlines()
        
        new_lines = []
        for node in nodes:
            for line in lines:
                if node not in line:
                    new_lines.append(line)

        with open(host_file, 'a') as w:
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

    print("old_job_nodes_dict: ", old_job_nodes_dict)
    print("new_job_nodes_dict: ", new_job_nodes_dict)

    # jobs
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
                add_job(newjob, new_job_nodes_dict[newjob], job_info_dict)

    # adjust the node
    overlappedJobs = utils.get_lists_overlap(newjobs, oldjobs)

    print("check on node level")
    # for each job check node changes
    for job in overlappedJobs:
        old_nodes = old_job_nodes_dict[job]
        new_nodes = new_job_nodes_dict[job]
        # nothing changed skip
        if set(old_nodes) == set(new_nodes):
            continue
        else:
            # pid = -1 means this job never launched before
            # len(old_nodes) == 0 means the no node assigned for this job
            if job_info_dict[job].pid == -1 and len(old_nodes) == 0:
                add_job(job, new_job_nodes_dict[job], job_info_dict)
                continue
            
            # job existed adjust nodes only
            intersectionNodes = utils.get_lists_overlap(old_nodes, new_nodes)
            addNodes =list(set(new_nodes) - set(intersectionNodes))
            print("addnodes ", addNodes)
            add_nodes_for_job(jobname=job, nodes=addNodes)
            delNodes = list(set(old_nodes) - set(intersectionNodes))
            print("deletenodes", delNodes)
            del_nodes_for_job(jobname=job, nodes=delNodes)
