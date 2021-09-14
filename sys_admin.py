import utils

def get_cluster_nodes():
    """The system will offer a api for getting the
        avaliable resources

    Returns:
        list: a list of avaliable nodes
    """
    if utils.is_theta_cluster():
        nodelist = ["thetagpu11","thetagpu13","thetagpu16","thetagpu17"]
        #nodelist = ["thetagpu02","thetagpu03","thetagpu04","thetagpu06",
        # "thetagpu08","thetagpu09","thetagpu10","thetagpu11","thetagpu13",
        # "thetagpu15","thetagpu16","thetagpu17","thetagpu18","thetagpu19",
        # "thetagpu21", "thetagpu22"]

    else:
        nodelist = ["node01", "node02", "node03", "node04"]
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
