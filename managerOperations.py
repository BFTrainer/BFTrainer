import os
from msgOperations import MSGOperations
from subprocess import Popen
import utils
import psutil
import stat

NUM_OF_GPUs_PER_NODE = 8
WORKING_DIR = utils.working_dir()

def create_working_directory():
    work_dir = os.path.exists(WORKING_DIR)
    if not work_dir:
        os.mkdir(WORKING_DIR)

def add_job(jobname, nodes, job_info_dict):
    print("Add job was called")
    print("jobname %s on nodes %s" %(jobname, nodes))
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
        print("command", command)
        p = Popen(command, shell=True, env=myenv, stdout=out, stderr=err)

        hvdrunParentPid = p.pid
        fp = psutil.Process(hvdrunParentPid)

        hvdpid = fp.children()[0].pid
        print("new job launch success and the hvdpid is:", hvdpid)

        # 3. update process id to `jobInfoDict`
        jobItem = job_info_dict[jobname]
        jobItem.pid = hvdpid

def create_discovery_file(jobname, nodes):
    discovery_path = os.path.join(WORKING_DIR,"discover_host_" + jobname + ".sh")
    with open(discovery_path, 'w') as w:
        w.write("#!/bin/bash\n\n")
        w.write("echo localhost:0\n")  # dummy computing node (the purpose for this line is for launch 1 node task at initial)
        for node in nodes:
            w.write("echo " + node + ":" + str(NUM_OF_GPUs_PER_NODE) + "\n")

    # grant host file executable permission
    st = os.stat(discovery_path)
    os.chmod(discovery_path, st.st_mode | stat.S_IRWXO | stat.S_IRWXG | stat.S_IRWXU)
    return discovery_path

def generate_command(discover_file_path, jobname, job_info_dict):
    print("jobname type", type(jobname))
    scriptPath = job_info_dict[jobname].path
    command = "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/horovodrun -np 1 --host-discovery-script " + discover_file_path + " python " + scriptPath + " --jobname " + jobname
    return command

def del_job(jobname, job_info_dict):
    print("del job called")
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
    print("add nodes for job %s called" % jobname)
    print("added nodes: ", nodes)    

    # discover host file
    discover_file_path = os.path.join(WORKING_DIR, "discover_host_" + jobname + ".sh")
    
    # Write node info to discover file
    if os.path.exists(discover_file_path):
        with open(discover_file_path, 'a') as w:
            for node in nodes:
                w.write("echo " + node + ":" + str(NUM_OF_GPUs_PER_NODE) + "\n")

def is_line_contain_delete_nodes(line, nodes):
    flag = False
    for node in nodes:
        if node in line:
            flag = True
            break
    return flag

def del_nodes_for_job(jobname, nodes):
    print("delete node for job %s called" % jobname)
    print("delete nodes: ", nodes)

    # del host from corresponding hostfile
    discover_file_path = os.path.join(WORKING_DIR, "discover_host_" + jobname + ".sh")
    if os.path.exists(discover_file_path):
        # Read the file filtered out the delete node line and put it back
        lines = []
        with open(discover_file_path, 'r') as r:
            lines = r.readlines()
        
        # filter out deleted nodes
        new_lines = []
        for line in lines:
            if is_line_contain_delete_nodes(line, nodes) == False:
                new_lines.append(line)

        # write back
        with open(discover_file_path, 'w') as w:
            for line in new_lines:
                w.write(line)

def adjust_jobs_and_nodes_by_maps_diff(new_map, old_map, job_info_dict):
    """Make adjustment by comparing two maps differences

    Args:
        newMap (dataframe): the input dataframe from optimizer
    """
    print("adjust_nodes_by_map(newmap, oldmap, jobInfoDict) called")

    # map to dict
    old_job_nodes_dict = utils.get_job_nodes_mapping_from(old_map)
    new_job_nodes_dict = utils.get_job_nodes_mapping_from(new_map)

    # Adjustment on job level
    oldjobs = list(old_job_nodes_dict.keys())
    newjobs = list(new_job_nodes_dict.keys())

    oldjobs.sort()
    newjobs.sort()

    # Job level adjustment
    if oldjobs != newjobs: # jobs changed
        for oldjob in oldjobs:
            if oldjob not in newjobs: # job in old not in new (del this job)
                del_job(oldjob)
        
        for newjob in newjobs:
            if newjob not in oldjobs: # job in new not in old (add this job)
                add_job(newjob, new_job_nodes_dict[newjob], job_info_dict)
    
    # Node level adjustment
    overlappedJobs = utils.get_lists_overlap(newjobs, oldjobs)

    # for each job check node changes
    # Only jobs in both old and new(overlabppedJobs) have node level changes

    # TODO:
    # Previously we only consider the extra node coming
    # Not considered that swich node between job
    # For switching nodes between running jobs, you need to make sure the del node 
    # happen first and then add node action.
    # For this part here *extra care is needed*

    for job in overlappedJobs:
        old_nodes = old_job_nodes_dict[job]
        new_nodes = new_job_nodes_dict[job]
        
        # check diff
        if set(old_nodes) != set(new_nodes):
            # pid = -1 means this job never launched before
            # len(old_nodes) == 0 means the no node assigned for this job
            if job_info_dict[job].pid == -1 and len(old_nodes) == 0:
                add_job(job, new_job_nodes_dict[job], job_info_dict)
                continue

            # job existed adjust nodes only
            intersectionNodes = utils.get_lists_overlap(old_nodes, new_nodes)
            addNodes =list(set(new_nodes) - set(intersectionNodes)) # node to be added
            if addNodes:
                add_nodes_for_job(jobname=job, nodes=addNodes)
            
            delNodes = list(set(old_nodes) - set(intersectionNodes)) # node to be deleted
            if delNodes:
                del_nodes_for_job(jobname=job, nodes=delNodes)
