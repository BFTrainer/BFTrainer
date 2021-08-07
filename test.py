from itertools import groupby
from time import time
from utils import UDP_Msg

m1 = UDP_Msg(address="55", id=1, time=12.00, rank_size=8, credit=11)
m2 = UDP_Msg(address="56", id=1, time=12.00,rank_size=8, credit=12)

msg_items = [m1, m2]

group_items = groupby(msg_items, lambda x: x.address)

group_dict = {}
for key, group in group_items:
    print("key", key)
    print("group", list(group))
    print("group", list(group))
    #group_dict[key] = list(group)

#print(group_dict)
