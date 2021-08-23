import os

# A data structure for hold job information
class JobInfo:
    def __init__(self, GUID, pid, max, min, N, O, resUp, resDown, path):
        self.GUID = GUID
        self.pid = pid
        self.max = max
        self.min = min
        self.N = N
        self.O = O
        self.resUp = resUp
        self.resDown = resDown
        self.path = path
