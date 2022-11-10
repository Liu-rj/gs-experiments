#!/bin/bash
cd ./fastgcn
./run.sh > outputs/log_all
cd ../ladies
./run.sh > outputs/log_all
cd ../pinsage
./run.sh > outputs/log_all