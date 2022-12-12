#!/bin/bash
cd ./shadowgnn
bash run.sh 1 > exp1206.log 2>exp1206.err
cd ../graphsaint
bash run.sh 1 > exp1206.log 2>exp1206.err
cd ../graphsage
bash run.sh 1 > exp1206.log 2>exp1206.err

