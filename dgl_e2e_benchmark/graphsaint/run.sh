#!/bin/bash
# cuda, dgl ad-hoc
# python train_minibatch.py --dataset=reddit --device cuda --sample-mode=ad-hoc
# python train_minibatch.py --dataset=products --device cuda --sample-mode=ad-hoc
# python train_minibatch.py --dataset=papers100m --device=cuda --use-uva --sample-mode=ad-hoc

# # cuda, dgl fine-grained
# python train_minibatch.py --dataset=reddit --device cuda --sample-mode=fine-grained
# python train_minibatch.py --dataset=products --device cuda --sample-mode=fine-grained
# python train_minibatch.py --dataset=papers100m --device=cuda  --use-uva --sample-mode=fine-grained

# # dgl cpu
# python train_minibatch.py --dataset=reddit --device cpu --num-workers=8 --sample-mode=ad-hoc
# python train_minibatch.py --dataset=products --device cpu --num-workers=8 --sample-mode=ad-hoc
# python train_minibatch.py --dataset=papers100m --device=cpu --num-workers=2 --use-uva 

# # matrix fused
# python train_minibatch.py --dataset=reddit --device cuda --sample-mode=matrix-fused
# python train_minibatch.py --dataset=products --device cuda --sample-mode=matrix-fused

# python train_minibatch.py --dataset=papers100m --device cuda  --use-uva --sample-mode=matrix-fused
# python train_minibatch.py --dataset=friendster --device=cuda --use-uva --sample-mode=ad-hoc
python train_minibatch.py --dataset=papers100m --device=cuda --use-uva --sample-mode=matrix-fused  --num-epoch=3
python train_minibatch.py --dataset=papers100m --device=cuda --use-uva --sample-mode=matrix-nonfused --num-epoch=3


# # matrix nonfused
# python train_minibatch.py --dataset=reddit --device cuda  --sample-mode=matrix-nonfused
# python train_minibatch.py --dataset=products --device cuda --sample-mode=matrix-nonfused
# python train_minibatch.py --dataset=papers100m --device cuda  --use-uva --sample-mode=matrix-nonfused

#nsys profile -f true -o shadow_dgl_profile python train_minibatch.py --dataset=friendster --device=cuda --use-uva --sample-mode=ad-hoc --samples=5,5 --num-epoch=1
nsys profile -f true -o shadow_fused_profile python train_minibatch.py --dataset=friendster --device=cuda --use-uva --sample-mode=matrix-fusedv2 --samples=5,5 --num-epoch=1
nsys profile -f true -o shadow_nonfused_profile python train_minibatch.py --dataset=friendster --device=cuda --use-uva --sample-mode=matrix-nonfused --samples=5,5 --num-epoch=1


