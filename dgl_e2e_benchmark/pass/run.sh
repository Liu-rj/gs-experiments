#!/bin/bash
# python train_matrix.py --dataset=reddit
python train_matrix.py --dataset=products
python train_matrix.py --dataset=papers100m --device=cpu --use-uva

# python train_dgl.py --dataset=reddit
# python train_dgl.py --dataset=products
# python train_dgl.py --dataset=papers100m --device=cpu --use-uva
