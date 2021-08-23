import utils

def get_avaliable_nodes_from_system():
    """The system will offer a api for getting the
        avaliable resources

    Returns:
        list: a list of avaliable nodes
    """
    if utils.is_theta_cluster():
        nodelist = ["thetagpu12","thetagpu13","thetagpu18","thetagpu21"]
    else:
        nodelist = ["node07", "node08", "node09", "node10"]
    return nodelist

def monitor_hvd_processes():
    # trigger job change
    pass

def monitor_nodes_status():
    # trigger node change
    pass

def job_information_from_client():
    # trigger job information change
    pass
