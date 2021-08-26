import argparse
from manager import Manager

parser = argparse.ArgumentParser(description="User need to submit job informations")
parser.add_argument('--min', type=int, required=True, help='min num of nodes')
parser.add_argument('--max', type=int, required=True, help="max num of nodes")
parser.add_argument('--N', type=int, required=True, nargs='+', help='num of nodes for scaling')
parser.add_argument('--O', type=float, required=True, nargs='+', help='objective ratio rate')
parser.add_argument('--res_up', type=int, required=True, help="scale up overhead")
parser.add_argument('--res_dw', type=int, required=True, help="scale down overhead")
parser.add_argument('--path', required=True, help="execute script path")
args = parser.parse_args()

def main():
    m = Manager(max_parallel=10, monitor_gap=10)
    id = m.submit_job(min=args.min, max=args.max, N=args.N, 
                    O=args.O, res_up=args.res_up, res_dw=args.res_dw, path=args.path)
    print("Job submitted! GUID:", str(id))

    '''
    # previous job submit
    id = m.submit_job(min=1, max=5, Ns=[1, 2, 3, 4, 5], 
                    Os=[1, 1.8, 2.6, 3.4, 4.2], res_up=3, res_dw=1, path="train.py")    
    '''

if __name__ == "__main__":
    main()
