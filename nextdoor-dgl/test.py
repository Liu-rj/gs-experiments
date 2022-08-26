import torch
import time
from graphsage_sampler import *
import load_graph

torch.ops.load_library("./nextdoor/build/libnextdoor.so")

FILE_PATH = "/home/ubuntu/NextDoorEval/NextDoor/input/reddit.data"
khop_sampler = torch.classes.my_classes.NextdoorKHopSampler(FILE_PATH)
khop_sampler.initSampling()

g, features, labels, n_classes, splitted_idx = load_graph.load_custom_reddit(FILE_PATH)
g = g.to('cuda')
seeds = g.nodes()
fanouts = [25, 10]


# accumulate_time = 0
# for i in range(10):
#     torch.cuda.synchronize()
#     start = time.time()
#     khop_sampler.sample()
#     torch.cuda.synchronize()
#     accumulate_time += time.time() - start
# print('sampling time:', accumulate_time / 10)


# final_samples = khop_sampler.finalSamples()
# print(final_samples, final_samples.shape, final_samples.dtype)
# print(khop_sampler.finalSampleLength())


acc_time = 0
for i in range(10):
    torch.cuda.synchronize()
    start = time.time()
    samples, blocks = neighborsampler_nextdoor(khop_sampler, seeds, fanouts)
    # blocks = neighborsampler_dgl(g, seeds, fanouts)
    torch.cuda.synchronize()
    acc_time += time.time() - start
print(acc_time / 10)
