#!/bin/bash
python train_matrix.py --dataset=reddit
python train_matrix.py --dataset=products
python train_matrix.py --dataset=papers100m --device=cpu --batchsize=2048 --samples=4000,4000 --use-uva

python train_dgl.py --dataset=reddit
python train_dgl.py --dataset=products
python train_dgl.py --dataset=papers100m --device=cpu --batchsize=2048 --samples=4000,4000 --use-uva

python train_dgl.py --dataset=reddit --device=cpu --num-workers=8
python train_dgl.py --dataset=products --device=cpu --num-workers=8
python train_dgl.py --dataset=papers100m --device=cpu --num-workers=8 --batchsize=2048 --samples=4000,4000