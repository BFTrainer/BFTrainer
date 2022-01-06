import os
import persistqueue

def submit_job_2_DBQueue(dbpath, jobString):
    """submit jobstring to persistqueue DB

    Args:
        jobString (string): jobstring

    Returns:
        int: job id in DB
    """
    q = persistqueue.SQLiteQueue(dbpath, auto_commit=True)
    return q.put(jobString)

def get_Job_from_DBQueue(dbpath):
    """get job string from persistqueue database

    Returns:
        string: jobstring
    """
    q = persistqueue.SQLiteQueue(dbpath, auto_commit=True)
    raw_dict = q.get(raw=True)
    jobid = raw_dict['pqid']
    jobstr = raw_dict['data']
    res = "GUID:" + str(jobid) + " " + jobstr
    return res

def get_DB_queue_len(dbpath):
    """Get the Queue length of the Database

    Returns:
        int: length of the queue in DB
    """
    q = persistqueue.SQLiteQueue(dbpath, auto_commit=True)
    return q.size
