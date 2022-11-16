#!/bin/bash
# cuda, dgl ad-hoc
python train_minibatch.py --dataset=reddit --device cuda --sample-mode=ad-hoc
python train_minibatch.py --dataset=products --device cuda --sample-mode=ad-hoc
python train_minibatch.py --dataset=papers100m --device=cuda --use-uva --sample-mode=ad-hoc

# cuda, dgl fine-grained
python train_minibatch.py --dataset=reddit --device cuda --sample-mode=fine-grained
python train_minibatch.py --dataset=products --device cuda --sample-mode=fine-grained
python train_minibatch.py --dataset=papers100m --device=cuda  --use-uva --sample-mode=fine-grained

# dgl cpu
python train_minibatch.py --dataset=reddit --device cpu --num-workers=8 --sample-mode=ad-hoc
python train_minibatch.py --dataset=products --device cpu --num-workers=8 --sample-mode=ad-hoc
python train_minibatch.py --dataset=papers100m --device=cpu --num-workers=2 --use-uva 
