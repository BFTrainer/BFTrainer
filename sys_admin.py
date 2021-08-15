
def get_avaliable_nodes_from_system():
    """The system will offer a api for getting the
        avaliable resources

    Returns:
        list: a list of avaliable nodes
    """
    #nodelist = ["thetagpu14","thetagpu15","thetagpu16","thetagpu20"]
    nodelist = ["node07", "node08", "node09", "node10"]
    nodelist.sort()
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
