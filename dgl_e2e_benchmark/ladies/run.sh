#!/bin/bash
python train.py --dataset=reddit
python train.py --dataset=products
python train.py --device=cpu --num-workers=12 --dataset=papers100m --batchsize=2048 --samples=4000,4000 --use-uva