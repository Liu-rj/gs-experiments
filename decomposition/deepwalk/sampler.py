import torch
import gs


def plain(A, seeds, num_steps):
    path = [seeds]
    for i in range(num_steps):
        subg = A._graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, False)
        subg = subg._CAPI_sampling(0, 1, False, gs._CSC, gs._CSC)
        seeds = subg._CAPI_get_coo_rows(False)
        path.append(seeds)
    return path


def inter_step_fusion(A, seeds, num_steps):
    path = [seeds]
    for i in range(num_steps):
        subg = A._graph._CAPI_fused_columnwise_slicing_sampling(seeds, 1, False)
        seeds = subg._CAPI_get_coo_rows(False)
        path.append(seeds)
    return path


def inter_layer_fusion(A, seeds, num_steps):
    path = A._graph._CAPI_random_walk(seeds, num_steps)
    return path
