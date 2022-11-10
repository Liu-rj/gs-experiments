#!/bin/bash
python model.py data/ml-1m
python model.py data/nowplaying_rs/

python model.py data/ml-1m --device=cpu --num-workers=8
python model.py data/nowplaying_rs/ --device=cpu --num-workers=8