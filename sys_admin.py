import utils

def get_cluster_nodes():
    """The system will offer a api for getting the
        avaliable resources

    Returns:
        list: a list of avaliable nodes
    """
    if utils.is_theta_cluster():
        nodelist = ["thetagpu12","thetagpu13","thetagpu18","thetagpu19"]
        #nodelist = ["thetagpu02","thetagpu03","thetagpu04","thetagpu06",
        #"thetagpu07","thetagpu08","thetagpu09","thetagpu10","thetagpu11",
        #"thetagpu14","thetagpu15","thetagpu16","thetagpu19","thetagpu20",
        #"thetagpu21", "thetagpu22"]

    else:
        nodelist = ["node07", "node08", "node09", "node10"]
    return nodelist

def is_nodes_belong_to_avaliable_nodes(nodes):
    ava_nodes = get_cluster_nodes()
    for node in nodes:
        if node not in ava_nodes:
            return False
    return True

def monitor_hvd_processes():
    # trigger job change
    pass

def monitor_nodes_status():
    # trigger node change
    pass

def job_information_from_client():
    # trigger job information change
    pass
