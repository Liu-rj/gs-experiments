#!/bin/bash
# python train_matrix.py --dataset=reddit --sampler=best
# python train_matrix.py --dataset=products --sampler=best
# python train_matrix.py --dataset=papers100m --device=cpu --use-uva --sampler=best

# python train_matrix.py --dataset=reddit --sampler=coo
python train_matrix.py --dataset=products --sampler=coo
python train_matrix.py --dataset=papers100m --device=cpu --use-uva --sampler=coo

# python train_dgl.py --dataset=reddit
# python train_dgl.py --dataset=products
# python train_dgl.py --dataset=papers100m --device=cpu --use-uva

# python train_dgl.py --dataset=reddit --device=cpu --num-workers=8
# python train_dgl.py --dataset=products --device=cpu --num-workers=8
# python train_dgl.py --dataset=papers100m --device=cpu --num-workers=8 --batchsize=2048 --samples=4000,4000