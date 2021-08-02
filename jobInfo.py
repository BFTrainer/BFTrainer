import os

# A data structure for hold job information
class JobInfo:
    def __init__(self, GUID, pid, max, min, Ns, Os, resUp, resDown, path):
        self.GUID = GUID
        self.pid = pid
        self.max = max
        self.min = min
        self.Ns = Ns
        self.Os = Os
        self.resUp = resUp
        self.resDown = resDown
        self.path = path
        # TODO:
        # consider darts scripts the python script may need some extra parameter
        # So leave this part here for future enhancement
