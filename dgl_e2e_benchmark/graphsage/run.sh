#!/bin/bash
# cuda, dgl ad-hoc
python train_dgl.py --dataset=reddit --device cuda --sample-mode=ad-hoc
python train_dgl.py --dataset=products --device cuda --sample-mode=ad-hoc
python train_dgl.py --dataset=papers100m --device=cuda --batchsize=2048 --use-uva --sample-mode=ad-hoc

# cuda, dgl fine-grained
python train_dgl.py --dataset=reddit --device cuda --sample-mode=fine-grained
python train_dgl.py --dataset=products --device cuda --sample-mode=fine-grained
python train_dgl.py --dataset=papers100m --device=cuda --batchsize=2048 --use-uva --sample-mode=fine-grained

# # dgl cpu
python train_dgl.py --dataset=reddit --device cpu --num-workers=8 --sample-mode=ad-hoc
python train_dgl.py --dataset=products --device cpu --num-workers=8 --sample-mode=ad-hoc
python train_dgl.py --dataset=papers100m --device=cpu --batchsize=2048 --num-workers=2 --use-uva 

# matrix nonfused
python train_matrix.py --dataset=reddit
python train_matrix.py --dataset=products
python train_matrix.py --dataset=papers100m --device=cpu --batchsize=2048 --use-uva

# matrix fused
python train_matrix.py --dataset=reddit --fused
python train_matrix.py --dataset=products --fused
python train_matrix.py --dataset=papers100m --device=cpu --batchsize=2048 --use-uva --fused
