#!/usr/bin/python

# this sample is wrong, the last item of N should be the same as `max`, similiar the first item should be `min`
# not necessary to give that many (N, O)
# this sample is valid for cases where user wants to optimize scaling efficiency
# python BFSub.py --min 1 --max 3 --N 1 2 3 4 5 --O 1 1.8 2.6 3.4 4.2 --res_up 3 --res_dw 1 --path train.py

# please consider to use this
python BFSub.py --min 1 --max 5 --N 1 2 3 4 5 --O 3500 5500 8500 11000 14000 --res_up 20 --res_dw 7 --path train.py