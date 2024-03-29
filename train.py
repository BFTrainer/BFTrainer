import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import time
from msgOperations import MSGClient

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=1,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=100000,
                    help='number of benchmark iterations')
parser.add_argument('--num-batches-per-commit', type=int, default=10,
                    help='number of batches per commit of the elastic state object')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--jobname', type = str, help='name of the job')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

hvd.init()
#print("debug mark: 1")

print("create udp client")
mo = MSGClient(address='10.201.4.145', port=5555)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())

cudnn.benchmark = True

# Set up standard model.
model = getattr(models, args.model)()


# By default, Adasum doesn't need scaling up learning rate.
def lr_scaler():
    return hvd.size() if not args.use_adasum else 1

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler())

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     op=hvd.Adasum if args.use_adasum else hvd.Average)

# Set up fixed fake data
data = torch.randn(args.batch_size, 3, 224, 224)
target = torch.LongTensor(args.batch_size).random_() % 1000
if args.cuda:
    data, target = data.cuda(), target.cuda()

def benchmark_step(state):
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

    #print("timestamp:%f worldsize:%d batch:%d" % (time.time(), hvd.size(), state.batch))
    state.batch += 1
    if state.batch == args.num_batches_per_commit:
        state.batch = 0
        state.commit()

def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')

log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

@hvd.elastic.run
def run_benchmark(state):
    # print("debug mark: 3")
   # Warm-up
    if not state.warm:
        log('Running warmup...')
        timeit.timeit(lambda: benchmark_step(state), number=args.num_warmup_batches)
        state.warm = True
        state.commit()

    # Benchmark
    if state.iter == 0:
        log('Running benchmark...')

    #print("debug mark: 4")
    for x in range(state.iter, args.num_iters):
        # time = timeit.timeit(lambda: benchmark_step(state), number=args.num_batches_per_iter)
        # tick = time.time()
        benchmark_step(state)
        # img_sec = args.batch_size * hvd.size() / tick
        # log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        if hvd.rank() == 0:
            mo.report(credit=args.batch_size * hvd.size(), rank_size = hvd.size(), jobname=args.jobname)

        state.img_secs.append(0)
        state.iter = x
        state.commit()

# Adjust learning rate on reset
def on_state_reset():
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_scaler()

# print("debug mark: 2")

state = hvd.elastic.TorchState(model, optimizer, img_secs=[], iter=0, batch=0, warm=False)
state.register_reset_callbacks([on_state_reset])
run_benchmark(state)

# Results
img_sec_mean = np.mean(state.img_secs)
img_sec_conf = 1.96 * np.std(state.img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
